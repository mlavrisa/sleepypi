import numpy as np
from scipy.signal import morlet2, correlate

def wavefinding_cwt(signal, widths, omega=5):
    output = np.empty((len(widths), len(signal)), dtype=np.complex128)
    for ind, width in enumerate(widths):
        # go for an odd window length about 8x the length of the width
        N = round(4 * width - 0.5) * 2 + 1
        N = np.min([N, len(signal)])
        wavelet_data = morlet2(N, width, omega)
        # using correlate instead of convolve
        output[ind] = correlate(
            signal.astype(np.complex128), wavelet_data, mode="same"
        ) * np.exp(-1.0j * omega * np.arange(len(signal)) / width)
    return output
