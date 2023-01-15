import gzip
import pickle
from time import time
from typing import Tuple

import numpy as np
from scipy.signal import sepfir2d
from picamera import PiCamera

from picamera.array import PiRGBAnalysis


class DetectMotion(PiRGBAnalysis):
    def __init__(
        self, camera: PiCamera, size: Tuple[int, int], lgth: int, gnorm: np.ndarray
    ):

        if int(time()) > 1824171564:
            print("Timestamps would overflow. Exiting.")
            raise OverflowError("Timestamps would overflow. Exiting.")

        self.camera = camera
        super().__init__(camera, size)
        # self.h = int(size[1] * 0.8) # the top 20% of the frame is where the curtain is
        self.h = 96
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
        self.data_stream = np.zeros((lgth, 16), dtype=np.float32)
        self.idx = 0
        self.jdx = 0
        self.timestamps = np.zeros(lgth, dtype=np.uint32)

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
        solnmat = (
            np.array([[-1.0, 3, -3, 1], [3, -3, -3, 3], [-3, -3, 3, 3], [1, 3, 3, 1]])
            / 8.0
        )
        # this ensures that the polynomial integrates to 0
        # this is so that exactly opposite motions square to the same number
        mapmat = np.array(
            [
                [self.n ** 3, 0, 0, 0],
                [0, self.n ** 2, 0, 0],
                [0, 0, self.n, 0],
                [0, -2.0 * self.n ** 2 / 3.0, 0, 0],
            ]
        )
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
        self.w_mask = np.real(np.copy(self.g_mask))
        self.gnorm = gnorm.astype(np.complex128)

        self.restart_flag = False
        self.file_loc = ""
        self.to_write = 0
        self.unwritten = True

        self.mags = np.zeros((self.h, self.w))
        self.args = np.zeros((self.h, self.w))

        self.video = np.zeros((lgth, self.h, self.w, 3), dtype=np.uint8)
        self.v_img = np.zeros((lgth, self.h, self.w, 2), dtype=np.uint8)

    def analyze(self, a: np.ndarray):
        self.video[self.idx] = a[self.cut :]
        fri = a[self.cut :].sum(axis=-1).astype(float)
        fri /= np.mean(fri).clip(min=1e-8)

        if self.unwritten:
            self.fhist[:] = fri
            self.fsum = fri * (self.n - 1.0)
            self.bhist[:] = self.fsum
            self.unwritten = False

        # keep a record of past frames
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
        # convert this to basis function components (equal areas)
        self.wxyz = np.tensordot(self.wxyz_trans, self.abcd, axes=1)
        self.wxyz[:] *= self.w_mask

        mag = np.linalg.norm(self.wxyz, axis=0)

        # treating wxyz as a quaternion, square it
        # this maps -wxyz and wxyz to the same place
        # this then gives us a way to find the average motion
        # including parts of the image that are brightening or dimming in sync
        self.wxyz2[0] = np.square(self.wxyz[0]) - np.sum(
            np.square(self.wxyz[1:]), axis=0
        )
        self.wxyz2[1:] = self.wxyz[1:] * 2.0 * self.wxyz[0:1]
        np.mean(self.wxyz2, axis=(1, 2), out=self.meanvec)
        # meanvec is the average squared quaternion direction - i.e. it finds the
        # movement vector which represents the most prominent breathing-like motion
        norm = np.linalg.norm(self.meanvec, axis=0, keepdims=True).clip(1e-8)
        self.meanvec /= norm
        # dots is a mask for those parts of the image which are moving in sync
        self.dots = np.clip(
            np.tensordot(self.meanvec, self.wxyz2, axes=1) / mag.clip(1e-8), 0.0, None
        )

        # calculate the gradient of the intercept
        gx = sepfir2d(self.abcd[3], self.gr, self.sm)
        gy = sepfir2d(self.abcd[3], self.sm, self.gr)
        self.gsum = gx + 1.0j * gy
        self.gsum *= self.g_mask

        # finally, calculate the velocity
        denom = np.real(np.sum(self.gsum * np.conj(self.gsum) * self.dots))
        if np.abs(denom) == 0.0:
            denom = 1e-8
        # normally velocity would use -dt * grad, but up is in -ve y direction
        v_img = self.abcd[2] * np.conj(self.gsum) * self.dots
        v = np.sum(v_img) / denom

        # calculate average light position + average movement location
        posmag = np.sum(
            np.arange(self.h)[:, None] * mag + 1.0j * np.arange(self.w)[None, :] * mag
        ) / np.sum(mag).clip(1e-8)
        pos = np.sum(
            np.arange(self.h)[:, None] * self.abcd[3]
            + 1.0j * np.arange(self.w)[None, :] * self.abcd[3]
        ) / np.sum(self.abcd[3]).clip(1e-8)

        np.abs(v_img, out=self.mags)
        self.args = np.angle(v_img)

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
        self.data_stream[self.idx, 10] = denom.astype(np.float32)
        self.data_stream[self.idx, 11] = np.real(norm[0])
        self.data_stream[self.idx, 12:] = self.meanvec

        # save velocity image
        self.v_img[self.idx, ..., 0] = (self.mags * 64.0).clip(max=255).astype(np.uint8)
        self.v_img[self.idx, ..., 1] = (self.args * 40.743665 + 128.0).astype(np.uint8)

        # timestamps are good until Oct 22 2027
        self.timestamps[self.idx] = round((time() - 1609459200.0) * 20.0)

        # if self.jdx % 40 == 0:
        #     print(self.jdx, self.restart_flag)

        # always increment jdx
        self.jdx += 1

        if self.restart_flag:
            # print("restart triggered")
            if not self.file_loc:
                raise RuntimeError("Must set file location for writing data")
            write_stream = (
                self.data_stream[: self.idx],
                self.timestamps[: self.idx],
            )
            write_vid = (
                self.video[: self.idx],
                self.v_img[: self.idx],
            )
            with gzip.open(self.file_loc + "-data.gz", "wb") as f:
                pickle.dump(write_stream, f)
            with gzip.open(self.file_loc + "-video.gz", "wb") as f:
                pickle.dump(write_vid, f)
            self.idx = 0
            self.file_loc = ""
            self.restart_flag = False
        else:
            self.idx += 1
