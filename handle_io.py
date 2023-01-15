import asyncio
from datetime import datetime
from os import makedirs
from os.path import exists

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from matplotlib.cm import magma
import vlc

import RPi.GPIO as gpio
from image_analysis import DetectMotion
from picamera import PiCamera


def datstr(timestamp=None) -> str:
    if timestamp is None:
        dt = datetime.now()
    else:
        dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%y%m%d_%H%M%S")


class OutPin:
    def __init__(self, index, state) -> None:
        self.index = index
        self.state = state

        self.setup()

    def setup(self):
        gpio.setup(self.index, gpio.OUT)
        gpio.output(self.index, self.state)

    def update_state(self, state):
        if self.state == state:
            return

        self.state = state
        gpio.output(self.index, self.state)


class PWMPin(OutPin):
    def __init__(self, index, state, freq) -> None:
        self.freq = freq
        self.pwm = None
        super().__init__(index, state)

    def setup(self):
        super().setup()
        self.pwm = gpio.PWM(self.index, self.freq)


class DataHandler:
    def __init__(self, home, drive_folder_id, ss_id) -> None:
        self.dr_service = None
        self.sh_service = None
        self.home = home
        self.drive_folder_id = drive_folder_id
        self.latest_subfolder_id = ""
        self.ss_id = ss_id

        self.build_services()

    def save_survey(self, data, started) -> None:
        body = {
            "values": [
                [
                    datstr(started),
                    datstr(),
                    data["headache"],
                    data["asthma"],
                    data["restful"],
                    data["light_sigs"],
                    data["food_time"],
                    data["alcohol"],
                    data["melatonin"],
                ]
            ]
        }
        query = (
            self.sh_service.spreadsheets()
            .values()
            .append(
                spreadsheetId=self.ss_id,
                range="A2:D",
                insertDataOption="INSERT_ROWS",
                valueInputOption="RAW",
                body=body,
            )
        )
        result = query.execute()
        updated = result.get("updates").get("updatedCells")
        if updated is None or not updated or updated < 1:
            print("Something went wrong while writing to the sheet!")

    def build_services(self):
        SCOPES = ["https://www.googleapis.com/auth/drive"]

        creds = None
        if exists(self.home + "token.json"):
            creds = Credentials.from_authorized_user_file(
                self.home + "token.json", SCOPES
            )
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.home + "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(self.home + "token.json", "w") as token:
                token.write(creds.to_json())

        self.dr_service = build("drive", "v3", credentials=creds)
        self.sh_service = build("sheets", "v4", credentials=creds)

    def create_folder(self, date_string):
        # if directory exists, return
        print(date_string, self.home + "sleepypi/run" + date_string)
        if exists(self.home + "sleepypi/run" + date_string):
            return

        # create the directory locally
        makedirs(self.home + "sleepypi/run" + date_string)
        # make the same directory in google drive
        folder_metadata = {
            "name": f"run{date_string}",
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [self.drive_folder_id],
        }
        drive_folder = (
            self.dr_service.files().create(body=folder_metadata, fields="id").execute()
        )

        self.latest_subfolder_id = drive_folder.get("id")

    def put_request(self, metadata, media):
        request = self.dr_service.files().create(
            body=metadata, media_body=media, fields="id"
        )
        response = None
        try:
            while response is None:
                status, response = request.next_chunk(num_retries=10)
        except HttpError as e:
            print(e)
            print(metadata)
            return False
        return response

    def save_segment(
        self, cam: PiCamera, detect_motion: DetectMotion, started_str: str
    ) -> None:

        ds = datstr()

        # take a capture
        cam.capture(
            f"{self.home}sleepypi/run{started_str}/{ds}.jpg",
            use_video_port=True,
            resize=(640, 480),
        )

        # send a signal to restart and where it should save
        detect_motion.restart_flag = True
        detect_motion.file_loc = f"{self.home}sleepypi/run{started_str}/{ds}"
        cam.wait_recording(0.225)

        return ds

    def upload_segment(self, started_str: str, ds: str):
        # if there was an error in starting the service, it will be None
        if self.dr_service is None:
            print("service is None")
            return

        capture_path = f"{self.home}sleepypi/run{started_str}/{ds}.jpg"
        gzip_path = f"{self.home}sleepypi/run{started_str}/{ds}-data.gz"

        capture_metadata = {
            "name": f"{ds}.jpg",
            "parents": [self.latest_subfolder_id],
        }
        gzip_metadata = {
            "name": f"{ds}-data.gz",
            "parents": [self.latest_subfolder_id],
        }

        capture_media = MediaFileUpload(
            capture_path, mimetype="image/jpeg", resumable=True
        )
        gzip_media = MediaFileUpload(
            gzip_path, mimetype="application/gzip", resumable=True
        )

        capture_file = self.put_request(capture_metadata, capture_media)
        gzip_file = self.put_request(gzip_metadata, gzip_media)

        # TODO: If these were successful, mark the local files for deletion after
        # processing.
        # ALTERNATIVELY: just save them locally, process them all at the end,
        # also locally. As they successfully upload, delete them locally. Maybe add a
        # periodically blinking light to indicate that there is processing going on.
        # Basically any time anything happens other than periodic breathing, or not
        # being in bed, keep that footage, as well as a few seconds around it.
        # Somewhat worried that this will also include things like curtain flapping,
        # but that's an issue anyway.

        if not capture_file or not gzip_file:
            print(f"Error: was not able to upload both files created at {ds}")

    def upload_video(self, started_str: str):
        # if there was an error in starting the service, it will be None
        if self.dr_service is None:
            return

        vid_path = f"{self.home}sleepypi/run{started_str}/analysis-{started_str}.mp4"

        vid_metadata = {
            "name": f"analysis-{started_str}.mp4",
            "parents": [self.latest_subfolder_id],
        }

        vid_media = MediaFileUpload(vid_path, mimetype="video/mp4", resumable=True)

        vid_file = self.put_request(vid_metadata, vid_media)

        # TODO: If these were successful, mark the local files for deletion after
        # processing.
        # ALTERNATIVELY: just save them locally, process them all at the end,
        # also locally. As they successfully upload, delete them locally. Maybe add a
        # periodically blinking light to indicate that there is processing going on.
        # Basically any time anything happens other than periodic breathing, or not
        # being in bed, keep that footage, as well as a few seconds around it.
        # Somewhat worried that this will also include things like curtain flapping,
        # but that's an issue anyway.

        if not vid_file:
            print(f"Error: was not able to upload video file for {started_str}")


class IOHandler:
    def __init__(self, params) -> None:

        # set up the camera
        self.res = (1640, 1232)
        self.sz = (160, 128)  # would most likely break at other resolutions
        self.fr = 5
        self.cam = PiCamera()
        self.cam.framerate = self.fr
        self.cam.resolution = self.res

        # set up the pins
        gpio.setmode(gpio.BOARD)

        self.IR = OutPin(index=7, state=gpio.LOW)
        self.BUSY = OutPin(index=13, state=gpio.HIGH)
        self.R = PWMPin(index=15, state=gpio.LOW, freq=2000)
        self.G = PWMPin(index=16, state=gpio.LOW, freq=2000)
        self.B = PWMPin(index=18, state=gpio.LOW, freq=2000)

        # TODO: add temperature and light sensors

    def start_recording(self, detect_motion_output) -> None:
        print("starting the recording...")
        self.cam.start_recording(detect_motion_output, format="rgb")
        self.cam.wait_recording(0.1)
        self.cam.awb_mode = "shade"
        self.cam.exposure_mode = "night"

    def stop_recording(self) -> None:
        print("stopping the recording...")
        self.cam.stop_recording()

    def indicate_tracking(self) -> None:
        self.IR.update_state(gpio.HIGH)
        self.BUSY.update_state(gpio.LOW)

    def indicate_idle(self) -> None:
        self.IR.update_state(gpio.LOW)
        self.BUSY.update_state(gpio.LOW)

    def indicate_processing(self) -> None:
        self.IR.update_state(gpio.LOW)
        self.BUSY.update_state(gpio.HIGH)

    def start_rgb(self) -> None:
        pwm = (self.R.pwm, self.G.pwm, self.B.pwm)
        [pin.start(0.0) for pin in pwm]

    def change_rgb(self, rgb) -> None:
        pwm = (self.R.pwm, self.G.pwm, self.B.pwm)
        [pin.ChangeDutyCycle(max(0, min(100 * dc))) for pin, dc in zip(pwm, rgb)]

    def stop_rgb(self) -> None:
        pwm = (self.R.pwm, self.G.pwm, self.B.pwm)
        [pin.stop() for pin in pwm]


class Alarm:
    def __init__(self, io: IOHandler, params,) -> None:
        self.io = io
        self.nap_alarm_fade = params.nap_alarm_fade
        self.night_alarm_fade = params.night_alarm_fade
        self.nap_lights_fade = params.nap_lights_fade
        self.night_lights_fade = params.night_lights_fade
        self.lights_task = None
        self.alarm_task = None
        self.media_player = vlc.MediaPlayer(params.media_loc)

    def start_nap(self, dur) -> None:
        if self.alarm_task is not None:
            raise RuntimeError("Alarm has already been set, cancel it first.")
        self.alarm_task = asyncio.create_task(
            self.alarm_fn(
                max(0.0, dur - self.nap_alarm_fade), min(dur, self.nap_alarm_fade)
            )
        )
        self.lights_task = asyncio.create_task(
            self.sunrise_fn(
                max(0.0, dur - self.nap_lights_fade), min(dur, self.nap_lights_fade)
            )
        )

    def start_night(self, dur) -> None:
        if self.alarm_task is not None:
            raise RuntimeError("Alarm has already been set, cancel it first.")
        self.alarm_task = asyncio.create_task(
            self.alarm_fn(
                max(0.0, dur - self.night_alarm_fade), min(dur, self.night_alarm_fade)
            )
        )
        self.lights_task = asyncio.create_task(
            self.sunrise_fn(
                max(0.0, dur - self.night_lights_fade), min(dur, self.night_lights_fade)
            )
        )

    def cancel_alarm(self) -> None:
        if self.alarm_task is None:
            return
        self.alarm_task.cancel()
        self.lights_task.cancel()
        self.alarm_task = None
        self.lights_task = None

        self.io.stop_rgb()
        self.media_player.stop()

    async def alarm_fn(self, wait_dur, fade_dur) -> None:
        await asyncio.sleep(wait_dur)
        # currently fade is built into the mp3, so it's not actually possible to change
        # start 15 seconds in, way too quiet before that
        self.media_player.play()
        # self.media_player.set_time(15_000)
        await asyncio.sleep(self.media_player.get_length() / 1000 - 0.5)  # - 15.5)

    async def sunrise_fn(self, wait_dur, fade_dur) -> None:
        await asyncio.sleep(wait_dur)
        self.io.start_rgb()
        steps = 3  # TODO: Set this back to 100 or so
        dt = fade_dur / steps
        for idx in range(steps):
            await asyncio.sleep(dt)
            f = (idx + 1) / steps
            self.io.change_rgb(magma(f)[:3])
            print(magma(f))

