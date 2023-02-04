import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np

from utilities import Params, StateMachine, TCPServer
from handle_io import Alarm, DataHandler, IOHandler
from image_analysis import DetectMotion


async def main():
    params = Params()

    gnorm = np.load(params.base + "gnorm.npy")

    data = DataHandler(params.base, params.drive_folder, params.ss_id, params.capture)
    io = IOHandler(params)
    alarm = Alarm(io, params)

    anlz = DetectMotion(
        io.cam, params.size, int(params.active_update_dur * params.fr + 20), gnorm
    )

    state = StateMachine(io, alarm, params, data, anlz)
    server = TCPServer(params.port, params.host, state)

    while True:
        try:
            await state.timer_task
        except asyncio.CancelledError:
            # a message was received that requires a change in the timer
            # create a new folder locally and in google drive
            # set a new timer based on the state of the state machine
            if state.state == StateMachine.IDLE:
                state.set_timer(params.idle_update_dur)
            elif state.state in [StateMachine.ALARM, StateMachine.WATCH]:
                if state.last_state == StateMachine.IDLE:
                    # this is a new tracking period, set a brand new timer
                    state.set_timer(params.active_update_dur)
                else:
                    # coming from a different tracking state, keep the same timer
                    rem = state.next_timer - time()
                    state.set_timer(rem, is_new=False)
        else:
            # timer ran out all by itself, save data if needed and start a new timer
            if state.state == StateMachine.IDLE:
                state.set_timer(params.idle_update_dur)
            else:
                ds = data.next_segment(io.cam, anlz, state.started_str)
                # data.upload_segment(state.started_str, ds)
                state.set_timer(params.active_update_dur)


def exception_hook(exc_type, exc_value, exc_traceback):
    logging.error(
        "Uncaught Exception, at " + datetime.now().strftime("%y%m%d_%H%M%S"),
        exc_info=(exc_type, exc_value, exc_traceback),
    )


if __name__ == "__main__":
    sys.excepthook = exception_hook
    p = Path.cwd() / "error_log.txt"
    logging.basicConfig(filename=str(p.absolute()), level=logging.INFO)
    logging.captureWarnings(True)
    asyncio.run(main())
