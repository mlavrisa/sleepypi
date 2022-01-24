import os
import numpy as np
from matplotlib import cm

os.environ["FFMPEG_BINARY"] = "ffmpeg"
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


class VideoWriter:
    def __init__(self, filename, fps=5.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()

    def from_array(self, input_array, percentile=100.0):
        ndarray = np.copy(input_array)
        if ndarray.dtype == np.complex128:
            mags = np.abs(ndarray)
            mags = mags.clip(max=np.percentile(mags, percentile))
            angs = np.mod(0.5 * np.angle(ndarray) / np.pi, 1.0)
            max_mag = np.max(mags).clip(1e-8)
            col = cm.hsv(angs)[..., :3]
            ndarray = col * mags[..., None] / max_mag
        elif np.any(ndarray < 0.0):
            ndarray /= 2.0 * np.percentile(np.abs(ndarray), percentile).clip(1e-8)
            ndarray += 0.5
        else:
            ndarray /= np.percentile(ndarray, percentile).clip(1e-8)
        for idx in range(ndarray.shape[0]):
            self.add(ndarray[idx])