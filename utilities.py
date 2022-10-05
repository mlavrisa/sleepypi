import asyncio
from datetime import datetime, timedelta
from json import loads as read_json
from time import time
from os.path import exists

from handle_io import Alarm, DataHandler, IOHandler


def datstr() -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S")


class Params:
    def __init__(self) -> None:
        # TODO: None of these make sense
        self.nap_alarm_fade = 9.0
        self.night_alarm_fade = 10.0
        self.nap_lights_fade = 11.0
        self.night_lights_fade = 12.0
        self.idle_update_dur = 10.0
        self.active_update_dur = 6.0
        self.size = (160, 120)
        self.fr = 5
        if exists("C:/Users"):
            self.pi = False
            self.media_loc = "C:/Users/mtlav/Music/birdsong_for_alarm.mp3"
            self.base = "C:/Users/mtlav/Development/personal-projects/sleepypi/"
        else:
            self.pi = True
            self.media_loc = "/home/pi/Music/birdsong_for_alarm.mp3"
            self.base = "/home/pi/Pictures/"


class StateMachine:
    IDLE = "idle"
    ALARM = "alarm"
    WATCH = "tracking"

    def __init__(
        self,
        io_handler: IOHandler,
        alarm_handler: Alarm,
        params: Params,
        data_handler: DataHandler,
    ) -> None:
        self.io_handler = io_handler
        self.alarm_handler = alarm_handler
        self.data_handler = data_handler
        self.params = params
        self.state = StateMachine.IDLE
        self.last_state = StateMachine.IDLE
        self.next_alarm = -1
        self.started = -1
        self.next_timer = -1
        self.last_timer = -1
        self.started_str = ""
        self.set_timer(params.idle_update_dur)

    def set_timer(self, dur, is_new=True):
        self.update_dur = dur
        if is_new:
            self.last_timer = round(time())
        self.next_timer = round(time() + dur)
        self.timer_task = asyncio.create_task(wait_dur(self.update_dur))

    def process_message(self, response) -> None:
        cmd = response["cmd"]
        data = None if "data" not in response else response["data"]
        if cmd == "wake" and not self.state == StateMachine.IDLE:
            # save data and stop recording
            self.data_handler.save_survey(data, self.started)
            self.last_state = self.state
            self.state = StateMachine.IDLE
            self.next_alarm = -1
            self.started = -1
            self.timer_task.cancel()
            self.alarm_handler.cancel_alarm()
        elif cmd in ["alarm", "snooze", "smart", "watch"]:
            # is the alarm supposed to go off sooner than the next timer loop would go?
            # if yes, interrupt and update the timer loop
            # if no, fuggetaboutit, just update state machine.
            # also want to do the alarm at some point, but we can add that later
            self.alarm_handler.cancel_alarm()
            if self.last_state == StateMachine.IDLE:
                self.started = round(time())
                self.started_str = datstr()
            self.last_state = self.state
            if cmd == "watch":
                self.state = StateMachine.WATCH
                self.next_alarm = -1
            elif cmd == "alarm":
                self.state = StateMachine.ALARM
                # TODO: Fix below, this only gets the hour??? And then treats it like seconds
                alarm_time = round(float(data["time"].split(":")[0]))
                now = datetime.now()
                delta = (alarm_time - now.second) % 60  # TODO: get rid of % 60
                next_alarm = now + timedelta(seconds=delta)
                self.next_alarm = round(next_alarm.timestamp())
                self.alarm_handler.start_night(delta)
            elif cmd == "snooze":
                self.state = StateMachine.ALARM
                dur = float(data["duration"][:-1]) #  TODO: treats minutes as seconds
                next_alarm = time() + dur
                self.next_alarm = round(next_alarm)
                self.alarm_handler.start_nap(dur)
            elif cmd == "smart":
                self.state = StateMachine.ALARM
                dur = float(data["duration"][:-1]) #  TODO: treats minutes as seconds
                next_alarm = time() + dur
                self.next_alarm = round(next_alarm)
                self.alarm_handler.start_night(dur)
            self.timer_task.cancel()


class TCPServer:
    def __init__(self, port, host, state_machine: StateMachine) -> None:
        self.port = port
        self.host = host
        self.cb = state_machine.process_message
        self.server_task = asyncio.create_task(self.start_server())

    async def start_server(self):
        async def cb_internal(reader, _):
            data = await reader.read(1024)
            data_str = data.decode()
            print(data_str)
            self.cb(read_json(data_str))

        self.server = await asyncio.start_server(cb_internal, self.host, self.port)
        async with self.server:
            await self.server.serve_forever()


async def wait_dur(dur):
    await asyncio.sleep(dur)
    print("timer finished")
