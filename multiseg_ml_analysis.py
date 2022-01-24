# %%
import gzip
import pickle
from datetime import datetime
from glob import glob

import warnings

from video_writer import VideoWriter

warnings.filterwarnings("error")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
from matplotlib import cm
from scipy.ndimage import correlate1d
from scipy.signal import (
    correlate,
    cspline1d,
    find_peaks,
    cwt,
    ricker,
    medfilt,
    morlet2,
    hilbert,
)
from tqdm import trange

# 211107_015105 -- this is a good one
# 211102_003909 -- another goodie. Lots of sleep, some time not in bed at the beginning.
#               -- No wake up though.
# 211101_002730 -- Excellent. 5 Sleep cycles visible. One spot not flipped right.
dt = "220103_232249"

gl = sorted(glob(f"sleepypi/run{dt}/*.pkl.gz"))

streams = []
times = []

# get timezone offset for local START of night, so that DST is handled appropriately
uctdiff = datetime.strptime(dt, "%y%m%d_%H%M%S").astimezone().utcoffset()
tzoffset = (uctdiff.days * 86400 + uctdiff.seconds) * 1000  # timezone offset from utc

for idx in trange(len(gl)):
    with gzip.open(gl[idx], "rb") as f:
        p = pickle.load(f)
        data_stream, fhist, gz, fri, tstamps, video = p
        streams.append(data_stream)
        times.append(tstamps.astype(np.int64) * 50 + 1609459200000 + tzoffset)
        with VideoWriter(gl[idx][:-6] + "mp4") as vid:
            vid.from_array(video)

n = np.concatenate(streams, axis=0)
# convert times back to epoch time in milliseconds, then to np.datetime64 in ms
timestamps = np.concatenate(times, axis=0).astype("<M8[ms]")


