import gzip
import pickle
from contextlib import redirect_stdout
from datetime import datetime
from os import mkdir
from os.path import exists
from time import sleep, time
from traceback import print_exc

import numpy as np
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from picamera import PiCamera
from picamera.array import PiRGBAnalysis
from scipy.signal import sepfir2d


class DetectMotion(PiRGBAnalysis):
    def __init__(self, camera, size, len, gnorm):

        if int(time()) > 1824171564:
            print("Timestamps would overflow. Exiting.")
            raise OverflowError("Timestamps would overflow. Exiting.")

        super().__init__(camera, size)
        self.h = int(size[1] * 0.8)  # the top 20% of the frame is where the curtain is
        self.cut = size[1] - self.h
        self.w = size[0]
        self.sm = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        self.gr = np.array([-1.0, -2.0, 0.0, 2.0, 1.0])
        self.lp = self.sm - np.mean(self.sm)
        self.un = np.ones(5, dtype=float)
        self.n = 5
        self.k = self.n // 2
        self.q = self.k * (self.k + 1) * (2 * self.k + 1) / 3
        self.wlen = self.n * 4
        self.fhist = np.ones((self.wlen, self.h, self.w))
        self.mhist = np.ones((self.wlen, self.h, self.w))
        self.bhist = np.ones((self.wlen, self.h, self.w))
        self.sumsq = np.ones((self.h, self.w))
        self.fslp = np.zeros((self.h, self.w))
        self.fsum = np.ones((self.h, self.w))
        self.data_stream = np.zeros((len, 6))
        self.idx = 0
        self.jdx = 0
        self.timestamps = np.zeros(len, dtype=np.uint32)

        xs = np.array([-self.n, 0.0, self.n])[:, None]
        powers = np.arange(4)[None, ::-1]
        A = np.zeros((6, 4))
        A[:3] = np.float_power(xs, powers) * self.n
        A[3:, :-1] = np.float_power(xs, powers[:, 1:]) * powers[:, :-1] * self.q
        AtA = np.dot(A.T, A)
        self.abcd_trans = np.dot(np.linalg.inv(AtA), A.T)
        self.obs = np.zeros((6, self.h, self.w))
        self.abcd = np.zeros((4, self.h, self.w))

        # 3rd degree bernstein polynomial coefficients
        solnmat = np.array([
            [-1.0, 3, -3, 1],
            [3, -3, -3, 3],
            [-3, -3, 3, 3],
            [1, 3, 3, 1]
        ]) / 8.0
        # this ensures that the polynomial integrates to 0
        # this is so that exactly opposite motions square to the same number
        mapmat = np.array([
            [self.n ** 3, 0, 0, 0],
            [0, self.n ** 2, 0, 0],
            [0, 0, self.n, 0],
            [0, -2.0 * self.n ** 2 / 3.0, 0, 0]
        ])
        self.wxyz_trans = np.dot(np.linalg.inv(solnmat), mapmat)
        self.wxyz = np.zeros((4, self.h, self.w))
        self.wxyz2 = np.zeros_like(self.wxyz)
        self.meanvec = np.zeros((4,))
        self.dots = np.zeros_like(self.wxyz)

        # set up the velocity mask. Any points above the line connecting 120,0 and 0,360
        # But shifted
        self.g_mask = 1.0 - np.logical_or(
            29.0 / 95.0
            - np.linspace(0.0, 1.0, self.w)[None, :] * 1856.0 / 3591.0
            - np.linspace(0.0, 1.0, self.h)[:, None]
            > 0.0,
            25.5 * np.linspace(0.0, 1.0, self.w)[None, :]
            - 95.0 * np.linspace(0.0, 1.0, self.h)[:, None]
            > 0.0,
        ).astype(np.complex128)
        self.g_mask[:5] = 0.0
        self.g_mask[:, 136:] = 0.0
        self.w_mask = np.copy(self.g_mask).astype(float)
        self.gnorm = gnorm.astype(np.complex128)

        self.restart_flag = False
        self.to_write = 0
        self.unwritten = True

        self.video = np.zeros((400, self.h, self.w))

    def analyze(self, a: np.ndarray):
        fri = a[self.cut :].astype(float)
        fri /= np.mean(fri).clip(min=1e-8)

        if self.unwritten:
            self.fhist[:] = fri
            self.fsum = fri * (self.n - 1.0)
            self.bhist[:] = self.fsum
            self.unwritten = False

        # keep a record of past frames, and their squares
        self.fhist[self.jdx % self.wlen] = fri

        # fslp - used to calculate slope
        # fsum - used to calculate intercept
        # higher order polynomials are numerically unstable
        self.fslp -= self.fsum
        self.fslp += fri * (self.n // 2)
        self.fslp += self.fhist[(self.jdx - self.n) % self.wlen] * (self.n // 2)
        self.fsum += fri
        self.fsum -= self.fhist[(self.jdx - self.n + 1) % self.wlen]

        # store these values, need them later for cubic approximation
        self.mhist[self.jdx % self.wlen] = self.fslp
        self.bhist[self.jdx % self.wlen] = self.fsum

        # combine the observations needed for the cubic approximation
        self.obs[0] = self.bhist[(self.jdx - 2 * self.n) % self.wlen]
        self.obs[1] = self.bhist[(self.jdx - self.n) % self.wlen]
        self.obs[2] = self.fsum
        self.obs[3] = self.mhist[(self.jdx - 2 * self.n) % self.wlen]
        self.obs[4] = self.mhist[(self.jdx - self.n) % self.wlen]
        self.obs[5] = self.fslp

        # cubic polynomial with roughly the right slopes and intercepts
        self.abcd = np.tensordot(self.abcd_trans, self.obs, axes=1)
        # convert this to a basis function components (equal areas)
        self.wxyz = np.tensordot(self.wxyz_trans, self.abcd, axes=1)
        self.wxyz[:] *= self.w_mask

        mag = np.linalg.norm(self.wxyz, axis=0)

        # treating wxyz as a quaternion, square it
        # this maps -wxyz and wxyz to the same place
        # this then gives us a way to find the average motion
        # including parts of the image that are moving in opposite directions
        self.wxyz2[0] = np.square(self.wxyz[0]) - np.sum(np.square(self.wxyz[1:]), axis=0)
        self.wxyz2[1:] = self.wxyz[1:] * 2.0 * self.wxyz[0:1]
        np.mean(self.wxyz2, axis=(1, 2), out=self.meanvec)
        self.meanvec /= np.linalg.norm(self.meanvec, axis=0, keepdims=True).clip(1e-8)
        self.dots = np.clip(np.tensordot(self.meanvec, self.wxyz2, axes=1) / mag.clip(1e-8), 0.0, None)

        # calculate the gradient of the intercept
        gx = sepfir2d(self.abcd[3], self.gr, self.sm)
        gy = sepfir2d(self.abcd[3], self.sm, self.gr)
        self.gsum = gx + 1.0j * gy
        self.gsum *= self.g_mask

        # finally, calculate the velocity
        denom = np.sum(self.gsum * np.conj(self.gsum) * self.dots)
        if np.abs(denom) == 0.0:
            denom = 1e-8
        # normally velocity would use -dt * grad, but up is in -ve y direction
        v = np.sum(self.abcd[2] * np.conj(self.gsum) * self.dots) / denom

        # calculate average light position + average movement location
        posmag = np.sum(
            np.arange(self.h)[:, None] * mag + 1.0j * np.arange(self.w)[None, :] * mag
        ) / np.sum(mag).clip(1e-8)
        pos = np.sum(
            np.arange(self.h)[:, None] * self.abcd[3]
            + 1.0j * np.arange(self.w)[None, :] * self.abcd[3]
        ) / np.sum(self.abcd[3]).clip(1e-8)

        # resize and save all of the output data
        self.data_stream[self.idx, 0] = np.real(v)
        self.data_stream[self.idx, 1] = np.imag(v)
        self.data_stream[self.idx, 2] = np.real(pos)
        self.data_stream[self.idx, 3] = np.imag(pos)
        self.data_stream[self.idx, 4] = np.real(posmag)
        self.data_stream[self.idx, 5] = np.imag(posmag)
        self.data_stream[self.idx, 6] = self.camera.iso
        self.data_stream[self.idx, 7] = self.camera.analog_gain
        self.data_stream[self.idx, 8] = self.camera.digital_gain
        self.data_stream[self.idx, 9] = self.camera.exposure_speed

        # save first little bit of video
        if self.idx < min(400, self.data_stream.shape[0]):
            self.video[self.idx] = fri

        # timestamps are good until Oct 22 2027
        self.timestamps[self.idx] = round((time() - 1609459200.0) * 20.0)

        # always increment jdx
        self.jdx += 1

        if self.restart_flag:
            self.to_write = (
                np.copy(self.data_stream[: self.idx]),
                np.copy(self.fhist),
                np.copy(self.gsum),
                np.copy(fri),
                np.copy(self.timestamps[: self.idx]),
                np.copy(self.video),
            )
            self.idx = 0
            self.restart_flag = False
        else:
            self.idx += 1


def main(base, home, datstr):

    try:
        # get google drive credentials
        service = get_service(base)

        # make the same directory in google drive
        parent_folder_id = "18j7nLLgED25-dQU5wht1GU49oJC_I3Xv"
        folder_metadata = {
            "name": f"run{datstr}",
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_folder_id],
        }
        drive_folder = service.files().create(body=folder_metadata, fields="id").execute()
        drive_folder_id = drive_folder.get("id")
    except Exception as e:
        print("Google authentication failed. Running in offline mode.")
        print_exc()
        service = None
        drive_folder_id = ""

    gnorm = np.load(base + "gnorm.npy")

    # setup run parameters
    hrs = 9
    freq_pic = 4
    hrl = 3600
    extra_time = 100
    fr = 5
    sz = (160, 120)  # would most likely break at other resolutions

    # setup the camera
    cam = PiCamera()
    cam.framerate = fr
    cam.resolution = sz

    # start recording
    with DetectMotion(cam, sz, fr * (hrl + extra_time) // freq_pic, gnorm) as output:
        print("starting the recording...")
        cam.start_recording(output, format="rgb")
        cam.awb_mode = "shade"
        cam.exposure_mode = "night"
        per_started = time()

        # iterate - run for a while, take a capture, gzip the data
        for idx in range(hrs * freq_pic):
            curr_time = time()
            cam.wait_recording(hrl // freq_pic - 1 - (curr_time - per_started))
            datstr = datetime.now().strftime("%y%m%d_%H%M%S")
            cam.capture(f"{home}{datstr}.jpg", use_video_port=True)
            print(f"image {idx+1} taken")

            # send a signal to restart, wait a second, then gzip
            output.restart_flag = True
            per_started = time()
            cam.wait_recording(1)
            with gzip.open(f"{home}{datstr}.pkl.gz", "wb") as f:
                pickle.dump(output.to_write, f)

            # attempt to upload the files to google drive
            save_data_to_drive(drive_folder_id, home, datstr, service)

        cam.stop_recording()
    cam.close()
    print("finished!")


def save_data_to_drive(drive_folder_id, home, datstr, service):
    # if there was an error in starting the service, it will be None
    if service is None:
        return

    capture_path = f"{home}{datstr}.jpg"
    gzip_path = f"{home}{datstr}.pkl.gz"

    capture_metadata = {
        "name": f"{datstr}.jpg",
        "parents": [drive_folder_id],
    }
    gzip_metadata = {
        "name": f"{datstr}.pkl.gz",
        "parents": [drive_folder_id],
    }
    capture_media = MediaFileUpload(capture_path, mimetype="image/jpeg", resumable=True)
    gzip_media = MediaFileUpload(gzip_path, mimetype="application/gzip", resumable=True)
    capture_file = put_request(service, capture_metadata, capture_media)
    gzip_file = put_request(service, gzip_metadata, gzip_media)

    if not capture_file or not gzip_file:
        print(f"Error: was not able to upload both files created at {datstr}")


def put_request(service, metadata, media):
    request = service.files().create(body=metadata, media_body=media, fields="id")
    response = None
    try:
        while response is None:
            status, response = request.next_chunk(num_retries=10)
    except HttpError as e:
        print(e)
        print(metadata)
        return False
    return response


def get_service(home):
    SCOPES = ["https://www.googleapis.com/auth/drive"]

    creds = None
    if exists(home + "token.json"):
        creds = Credentials.from_authorized_user_file(home + "token.json", SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                home + "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(home + "token.json", "w") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


if __name__ == "__main__":

    # allow time for user to exit application before anything happens
    sleep(30)

    # make the home directory. Why it's in pictures, we'll never know.
    dt = datetime.now()
    datstr = dt.strftime("%y%m%d_%H%M%S")
    base = "/home/pi/Pictures/"
    home = f"{base}sleepypi/run{datstr}/"

    mkdir(home)

    with open(f"{home}{datstr}.log", "w") as stdout, redirect_stdout(stdout):
        try:
            main(base, home, datstr)
        except Exception as e:
            print_exc()
            quit()
