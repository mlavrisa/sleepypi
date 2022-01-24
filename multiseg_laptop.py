import gzip
import pickle
from glob import glob
from os.path import exists
from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sepfir2d
from scipy.signal.bsplines import cspline1d
from tqdm import trange

from video_writer import VideoWriter



class DetectMotion:
    def __init__(self, camera, size, len, gnorm):

        if int(time()) > 1824171564:
            raise OverflowError("Timestamps would overflow. Exiting.")

        # super().__init__(camera, size)
        # self.h = int(size[1] * 0.8)  # the top 20% of the frame is where the curtain is
        self.h = size[1]
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

        solnmat = np.array([
            [-1.0, 3, -3, 1],
            [3, -3, -3, 3],
            [-3, -3, 3, 3],
            [1, 3, 3, 1]
        ]) / 8.0
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

        self.video = np.zeros((500, self.h, self.w), dtype=np.complex128)

    # @profile
    def analyze(self, a: np.ndarray):
        fri = a.astype(float)  # (
        #     a[self.cut :]
        #     .reshape(self.h // self.rsz, self.rsz, self.w // self.rsz, self.rsz, 3)
        #     .sum(axis=(1, 3, 4))
        #     .astype(float)
        # )
        # what if we didn't divide by the mean?
        # this should help normalize the velocity for exceptionally poorly lit scenes
        # fri /= np.mean(fri).clip(min=1e-8)

        if self.unwritten:
            self.fhist[:] = fri
            self.fsum = fri * (self.n - 1.0)
            self.bhist[:] = self.fsum
            self.unwritten = False

        self.fhist[self.jdx % self.wlen] = fri

        self.fslp -= self.fsum
        self.fslp += fri * (self.n // 2)
        self.fslp += self.fhist[(self.jdx - self.n) % self.wlen] * (self.n // 2)
        self.fsum += fri
        self.fsum -= self.fhist[(self.jdx - self.n + 1) % self.wlen]

        self.mhist[self.jdx % self.wlen] = self.fslp
        self.bhist[self.jdx % self.wlen] = self.fsum

        self.obs[0] = self.bhist[(self.jdx - 2 * self.n) % self.wlen]
        self.obs[1] = self.bhist[(self.jdx - self.n) % self.wlen]
        self.obs[2] = self.fsum
        self.obs[3] = self.mhist[(self.jdx - 2 * self.n) % self.wlen]
        self.obs[4] = self.mhist[(self.jdx - self.n) % self.wlen]
        self.obs[5] = self.fslp

        self.abcd = np.tensordot(self.abcd_trans, self.obs, axes=1)
        self.wxyz = np.tensordot(self.wxyz_trans, self.abcd, axes=1)
        self.wxyz[:] *= self.w_mask

        mag = np.linalg.norm(self.wxyz, axis=0)

        # even = self.wxyz[0] * self.wxyz[3] + self.wxyz[1] * self.wxyz[2]
        # odd = self.wxyz[0] * self.wxyz[1] + self.wxyz[2] * self.wxyz[3]
        # phase = self.wxyz[0] + 1.0j * self.wxyz[1]#even + 1.0j * odd

        self.wxyz2[0] = np.square(self.wxyz[0]) - np.sum(np.square(self.wxyz[1:]), axis=0)
        self.wxyz2[1:] = self.wxyz[1:] * 2.0 * self.wxyz[0:1]
        np.mean(self.wxyz2, axis=(1, 2), out=self.meanvec)
        self.meanvec /= np.linalg.norm(self.meanvec, axis=0, keepdims=True).clip(1e-8)
        self.dots = np.clip(np.tensordot(self.meanvec, self.wxyz2, axes=1) / mag.clip(1e-8), 0.0, None)

        gx = sepfir2d(self.abcd[3], self.gr, self.sm)
        gy = sepfir2d(self.abcd[3], self.sm, self.gr)
        self.gsum = gx + 1.0j * gy
        self.gsum *= self.g_mask

        denom = np.sum(self.gsum * np.conj(self.gsum) * self.dots)
        if np.abs(denom) == 0.0:
            denom = 1e-8
        # normally velocity would use -dt * grad, but up is in -ve y direction
        v = np.sum(self.abcd[2] * np.conj(self.gsum) * self.dots) / denom

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
        # self.data_stream[self.idx, 4] = self.camera.iso
        # self.data_stream[self.idx, 5] = self.camera.analog_gain
        # self.data_stream[self.idx, 6] = self.camera.digital_gain
        # self.data_stream[self.idx, 7] = self.camera.exposure_speed
        # self.data_stream[self.idx, 8] = self.camera.brightness
        # self.data_stream[self.idx, 9] = self.camera.contrast

        # save first little bit of video
        if self.idx < min(500, self.data_stream.shape[0]):
            self.video[
                self.idx
            ] = self.dots#self.abcd[2] * np.conj(self.gsum) * self.dots

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


def main(base, file, vid, output):

    out_fn = file[:-7] + ".mp4"
    if not exists(out_fn):
        with VideoWriter(out_fn) as video:
            vis = np.copy(vid)
            vis /= np.max(vid)
            nfr = vid.shape[0]
            for idx in trange(nfr):
                video.add(vis[idx])

    # print("video")
    for idx in trange(vid.shape[0]):
        if idx == vid.shape[0] - 1:
            output.restart_flag = True
        output.analyze(vid[idx])

    with VideoWriter(file[:-7] + "_dots.mp4") as vid:
        vid.from_array(output.to_write[-1][5:], 99.8)

    return output.to_write[0]


if __name__ == "__main__":
    base = "C:/Users/mtlav/Development/personal-projects/sleepypi/sleepypi/run211210_235337/"

    files = sorted(glob(base + "*.pkl.gz"))
    velocities = []
    other_streams = []
    nfr, h, w = 400, 96, 160
    gm_norm = np.load("gnorm.npy")
    output = DetectMotion(None, (w, h), nfr, gm_norm)
    for file in files:
        with gzip.open(file, "rb") as f:
            data_stream, fhist, gsum, fri, tstamps, vid = pickle.load(f)
            other_streams.append(data_stream[:vid.shape[0]])
        velocities.append(main(base, file, vid, output))

    alldata = np.concatenate(other_streams, 0)
    v = np.concatenate(velocities)
    np.save(base + "velocities.npy", v)

    # _, ax = plt.subplots(1, 1)
    # ax.plot(alldata)
    # ax.legend([
    #     "real(v)",
    #     "imag(v)",
    #     "iso",
    #     "analog_gain",
    #     "digital_gain",
    #     "exposure_speed",
    #     "brightness",
    #     "contrast",
    # ])
    # plt.show()

    csvx = np.cumsum(v[:, 0])
    csvy = np.cumsum(v[:, 1])

    _, ax = plt.subplots(1, 1)
    ax.plot(v[:, 0])
    ax.plot(v[:, 1])
    ax.plot(alldata[:, 0], alpha=0.3)
    ax.plot(alldata[:, 1], alpha=0.3)
    plt.show()

    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(v[:, 0])
    ax1.plot(v[:, 1])
    ax2.plot(csvx)
    ax2.plot(csvy)
    ax1.vlines(np.arange(1, len(files)) * 400.0, -5e-3, 5e-3)
    ax2.vlines(np.arange(1, len(files)) * 400.0, -7e-2, 7e-2)
    ax3.plot(v[:, 2])
    ax3.plot(v[:, 3])
    ax3.plot(v[:, 4])
    ax3.plot(v[:, 5])
    plt.show()

    cvx = cspline1d(v[:, 0], 5.0)
    cvy = cspline1d(v[:, 1], 5.0)
    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(cvx)
    ax1.plot(cvy)
    ax2.plot(np.cumsum(cvx))
    ax2.plot(np.cumsum(cvy))
    ax1.vlines(np.arange(1, len(files)) * 400.0, -2.5e-3, 2.5e-3)
    ax2.vlines(np.arange(1, len(files)) * 400.0, -6e-2, 6e-2)
    plt.show()

    pass

