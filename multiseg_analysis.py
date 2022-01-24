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
import ruptures as rpt
from tqdm import trange


def smooth_and_norm_real(vals, winsize=95, smoothing=0.2, zcutoff=3.0):
    win = np.ones((winsize,))
    win_count = correlate(np.ones_like(vals), win)
    signal = np.copy(vals)
    init_rms = np.sqrt(correlate(np.square(signal), win) / win_count)
    init_cond = np.where(
        init_rms[: 1 - winsize] < init_rms[winsize - 1 :],
        init_rms[: 1 - winsize],
        init_rms[winsize - 1 :],
    )
    for _ in range(8):
        moving_rms = np.sqrt(correlate(np.square(signal), win).clip(0.0) / win_count)
        min_rms = np.minimum(moving_rms[: 1 - winsize], moving_rms[winsize - 1 :]).clip(
            1e-5
        )
        magclip = (
            zcutoff * min_rms / np.maximum(min_rms * zcutoff, np.abs(signal)).clip(1e-5)
        )
        signal *= magclip
    moving_avg = correlate(signal / min_rms, win) / win_count
    min_avg = np.where(
        init_cond,
        moving_avg[: 1 - winsize],
        moving_avg[winsize - 1 :],
    )
    min_avg = moving_avg[winsize // 2 : -(winsize // 2)]

    smoothed = cspline1d(signal / min_rms - min_avg, lamb=smoothing)

    return smoothed, vals / min_rms - min_avg


def get_rms(signal, winsize=95):
    win = np.ones((winsize,))
    win_count = correlate(np.ones_like(signal), win, mode="same")
    rms = np.sqrt(correlate(np.square(signal), win, mode="same") / win_count)
    return rms

def norm_complex(real, imag, winsize=95, smoothing=0.2, zcutoff=3.0):
    win = np.ones((winsize,))
    win_count = correlate(np.ones_like(real), win)
    signal = real + 1j * imag
    for _ in range(8):
        moving_rms = np.sqrt(correlate(np.square(np.abs(signal)), win) / win_count)
        min_rms = np.minimum(moving_rms[: 1 - winsize], moving_rms[winsize - 1 :])
        magclip = (
            zcutoff * min_rms / np.maximum(min_rms * zcutoff, np.abs(signal)).clip(1e-5)
        )
        signal *= magclip
    return (real + 1.0j * imag) / min_rms


def smooth_and_norm_complex_stitch(
    real_lead, imag_lead, real_lag, imag_lag, winsize=95, smoothing=0.2, zcutoff=3.0
):
    orig_ld_re = cspline1d(real_lead, lamb=smoothing)
    orig_ld_im = cspline1d(imag_lead, lamb=smoothing)
    orig_lg_re = cspline1d(real_lag, lamb=smoothing)
    orig_lg_im = cspline1d(imag_lag, lamb=smoothing)
    win = np.ones((winsize,))
    win_count = correlate(np.ones_like(real_lead), win)
    lead_signal = orig_ld_re + 1.0j * orig_ld_im
    lag_signal = orig_lg_re + 1.0j * orig_lg_im
    # TODO: pick a better way of stitching together, eg which is waviest?
    # TODO: also should do a moving average of each individual component and subtract it
    # currently there are jumps when it switches from one to the other. These suck.
    # TODO: if a component is NOT very wavy, should shrink it by, eg, 1/3
    # currently some components are very noisy in some places and this causes problems.
    lead_rms = np.sqrt(correlate(np.square(np.abs(lead_signal)), win) / win_count)
    lag_rms = np.sqrt(correlate(np.square(np.abs(lag_signal)), win) / win_count)
    signal = np.where(
        lead_rms[: 1 - winsize] < lag_rms[winsize - 1 :], lead_signal, lag_signal
    )
    orig_sig = np.copy(signal)
    for _ in range(8):
        moving_rms = np.sqrt(correlate(np.square(np.abs(signal)), win) / win_count)
        min_rms = np.minimum(moving_rms[: 1 - winsize], moving_rms[winsize - 1 :])
        magclip = (
            zcutoff * min_rms / np.maximum(min_rms * zcutoff, np.abs(signal)).clip(1e-5)
        )
        signal *= magclip
    return orig_sig / min_rms


def fit_linear_lead_and_lag(yi, winsize=155):
    k = winsize - 1
    sd = k * (k + 1) / 2
    sdsq = k * (k + 1) * (2 * k + 1) / 6
    win = np.ones(winsize)
    invdenom = 1.0 / (sdsq * winsize - sd ** 2)
    slp_kern = np.arange(winsize)
    fslp_lag = correlate(yi, slp_kern, mode="full")[winsize - 1:]
    fslp_lead = correlate(yi, -slp_kern[::-1], mode="full")[:1 - winsize]
    fsum = correlate(yi, win, mode="full")
    fsum_lead = fsum[:1 - winsize]
    fsum_lag = fsum[winsize - 1:]
    yi2 = np.square(yi)
    sumsq = correlate(yi2, win, mode="full")
    sumsq_lead = sumsq[:1 - winsize]
    sumsq_lag = sumsq[winsize - 1:]
    m_lag = (fslp_lag * winsize - sd * fsum_lag) * invdenom
    b_lag = (fsum_lag * sdsq - sd * fslp_lag) * invdenom
    m_lead = (fslp_lead * winsize + sd * fsum_lead) * invdenom
    b_lead = (fsum_lead * sdsq + sd * fslp_lead) * invdenom
    serr_lag = (
        np.square(b_lag) * winsize
        - 2 * b_lag * fsum_lag
        + 2 * b_lag * m_lag * sd
        + sumsq_lag
        - 2 * m_lag * fslp_lag
        + np.square(m_lag) * sdsq
    ) / winsize
    serr_lead = (
        np.square(b_lead) * winsize
        - 2 * b_lead * fsum_lead
        - 2 * b_lead * m_lead * sd
        + sumsq_lead
        - 2 * m_lead * fslp_lead
        + np.square(m_lead) * sdsq
    ) / winsize
    # var = sumsq + 4.0 * (winsize / sdsq - 1.0) * sqsum / sdsq
    # notice the clips - if variance is zero, r^2 is zero
    # r2 = 1.0 - serr / var.clip(1e-8)

    return m_lead * winsize, m_lag * winsize, b_lead, b_lag, np.sqrt(serr_lead), np.sqrt(serr_lag)


def compute_stats(xi, yi, winsize=255, mode="valid"):
    hw = winsize // 2
    hwmo = (winsize - 1) // 2
    win = np.ones((winsize,))
    xiyi = xi * yi
    xi2 = np.square(xi)
    yi2 = np.square(yi)
    win_xiyi = correlate(xiyi, win, mode=mode)
    win_xi2 = correlate(xi2, win, mode=mode)
    win_yi2 = correlate(yi2, win, mode=mode)
    m = win_xiyi / win_xi2.clip(1e-8)
    b = correlate(yi, win, mode=mode)
    idx = np.arange(xi.shape[0] - winsize + 1)[:, None] + np.arange(winsize)
    xwin = xi[idx]
    ywin = yi[idx]
    yhat = xwin * m[:, None] + b[:, None]
    sse = np.sum(np.square(ywin - yhat), axis=1)
    r2 = 1 - sse / win_yi2

    # angles = -np.arctan(m)
    # angles = np.concatenate(
    #     (np.full((hwmo,), angles[0]), angles, np.full((hw,), angles[-1]))
    # )
    # R = np.array([[np.cos(angles), -np.sin(angles)], [np.sin(angles), np.cos(angles)]])
    # input = np.stack((xi, yi), axis=0)[:, None, :]
    # res = np.sum(R * input, axis=1)[0]

    # res3 = res ** 3
    # win_res3 = correlate(res3, win, mode=mode)
    # skew = np.concatenate(
    #     (np.full((hw,), win_res3[0]), win_res3, np.full((hw,), win_res3[-1]))
    # )

    return b, m, r2  #, res, skew


def correlate_peaks(sig, start, length, min_shift, chunk):
    seg = sig[start : start + chunk]
    comp = sig[start + min_shift : start + length + min_shift]
    return correlate(comp, seg, mode="valid"), seg, comp


def get_turning_points(sig):
    deltas = sig[1:] - sig[:-1]
    turn_pts = deltas[1:] * deltas[:-1] <= 0.0
    curvature = correlate(
        sig, np.array([5.0, 0.0, -3.0, -4.0, -3.0, 0.0, 5.0]), mode="same"
    )

    peaks = np.argwhere(np.logical_and(deltas[:-1] > 0.0, turn_pts))
    troughs = np.argwhere(np.logical_and(deltas[:-1] < 0.0, turn_pts))

    # unlikely that a true breath would be faster than 3 seconds, ie 15 frames
    dist = 15
    peak_dists = peaks[1:] - peaks[:-1]
    trough_dists = troughs[1:] - troughs[:-1]

    # find the highest peak, eliminate any peaks too closeby, which means also
    # eliminating some troughs... but which ones... fuck, I have to think about this.
    while False:
        pass


def compute_r2(data, fit, winsize=255):
    hw = winsize // 2
    win = np.ones((winsize,))
    err = data - fit
    avgs = correlate(data, win, mode="same") / winsize
    sum_err2 = correlate(np.square(err), win, mode="same")
    idx = np.arange(data.shape[0])[:, None] + np.arange(winsize)
    data_ext = np.concatenate((np.zeros(hw), data, np.zeros(hw)), axis=0)
    sum_var = np.sum(np.square(data_ext[idx] - avgs[:, None]), axis=1)
    sum_var = np.var(data_ext[idx] - avgs[:, None], axis=1) * winsize
    r2 = 1 - sum_err2 / sum_var.clip(1e-5)

    return r2, err, sum_err2, sum_var


def get_findiff_curvature(data):
    curvature = np.zeros_like(data)
    curvature[1:-1] = data[:-2] + data[2:] - 2.0 * data[1:-1]
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    return curvature


def morlet_real(*args, **kwargs):
    return np.real(morlet2(*args, **kwargs))


def get_slopes(data):
    dx = data[:, 1:] - data[:, :-1]  # x right
    dy = data[1:] - data[:-1]  # y down
    de = data[1:, 1:] - data[:-1, :-1]  # y down, x right
    do = data[1:, :-1] - data[:-1, 1:]  # y down, x left

    tl = de[:-1, :-1]
    tc = dy[:-1, 1:-1]
    tr = do[:-1, 1:]
    rc = -dx[1:-1, 1:]
    br = -de[1:, 1:]
    bc = -dy[1:, 1:-1]
    bl = -do[1:, :-1]
    lc = dx[1:-1, :-1]

    res = np.zeros((8, data.shape[0], data.shape[1]))
    res[:, 1:-1, 1:-1] = np.stack((tl, tc, tr, rc, br, bc, bl, lc), axis=0)
    return res


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


def suppress_noise(real, imag):
    """Normalizes the traces, then compares them and weights the wavier one higher."""
    real_norm, _ = smooth_and_norm_real(real)
    imag_norm, _ = smooth_and_norm_real(imag)
    omega = 20.0
    fs = 5.0
    freqs = np.logspace(0.1, -1.4, 150)  # ~50-85 are breathing frequencies
    widths_morlet = omega * fs / (freqs[50:85] * 2 * np.pi)
    real_wave = wavefinding_cwt(real_norm, widths_morlet, omega)
    mags_real = np.sum(np.square(np.abs(real_wave)), axis=0)
    imag_wave = wavefinding_cwt(imag_norm, widths_morlet, omega)
    mags_imag = np.sum(np.square(np.abs(imag_wave)), axis=0)
    # clip at 4 - don't want to be making up bs waves
    ratio = np.square(mags_real / mags_imag).clip(0.25, 4.0)
    # normalize again after the ratio thing
    normed = norm_complex(real_norm * ratio, imag_norm / ratio)
    # these work pretty well I've found
    widths_peak = np.arange(4, 80) * 0.2
    cplx_peaks = cwt(normed, ricker, widths_peak, dtype=np.complex128)
    cplx_slopes = get_slopes(np.abs(cplx_peaks))
    peak_cond = np.sum(np.sign(cplx_slopes), axis=0) == 8.0
    peak_vals = cplx_peaks[peak_cond]
    arg = np.mod(np.angle(np.square(cplx_peaks)) * 0.5 / np.pi, 1.0)
    cols = cm.hsv(arg)[..., :3]
    mags = np.abs(cplx_peaks)
    mags *= 999.0 / np.max(mags)
    mags += 1.0
    cols *= np.log10(mags[..., None]) / 3.0
    cols[peak_cond, :] = 0.0

    amax = np.argmax(np.abs(cplx_peaks[:26]), axis=0).astype(float)
    amax_sm = np.around(cspline1d(medfilt(amax, 1023), 5000000)).astype(int)
    asym_trace = np.mean(np.abs(cplx_peaks[5:25]), axis=0)
    max_trace = np.abs(cplx_peaks[amax_sm, np.arange(amax.size)])

    # find the minima of both traces
    asym_min = np.logical_and(
        asym_trace[:-2] > asym_trace[1:-1], asym_trace[1:-1] < asym_trace[2:]
    )
    max_min = np.logical_and(
        max_trace[:-2] > max_trace[1:-1], max_trace[1:-1] < max_trace[2:]
    )
    asym_mean = np.mean(cplx_peaks[6:35], axis=0)  # empirical values
    asym_rot = np.angle(asym_mean[1:] / asym_mean[:-1])
    rot_conv = correlate(asym_rot, np.ones(3), mode="same")
    asym_pk_arg, *_ = find_peaks(
        np.abs(rot_conv), height=0.9, distance=4, prominence=0.2
    )
    asym_min_arg = np.argwhere(asym_min).squeeze()
    max_min_arg = np.argwhere(max_min).squeeze()
    # min_max_min = max_min_arg - 2
    # max_max_min = max_min_arg + 2
    # min_cond = np.any(
    #     np.logical_and(
    #         asym_min_arg[:, None] >= min_max_min[None, :],
    #         asym_min_arg[:, None] <= max_max_min[None, :],
    #     ),
    #     axis=1,
    # )
    # rem_args = asym_min_arg[min_cond] + 1
    rem_args = asym_pk_arg
    arg_diffs = rem_args[1:] - rem_args[:-1]
    mid_heights = max_trace[np.around(0.5 * (rem_args[1:] + rem_args[:-1])).astype(int)]

    # used for calculations inside
    nfold = np.sqrt(np.square(normed))
    nsq = np.abs(normed) * normed
    rotated = np.zeros_like(normed, dtype=np.complex128)

    # now we need to segment where the signal goes outside of the range (-3.0, 3.0)
    past_3 = np.argwhere(np.abs(normed) > 3.0).squeeze()
    num_outliers = len(past_3)
    outliers = np.zeros(num_outliers + 2, dtype=int)
    outliers[1:-1] = past_3
    outliers[-1] = len(normed) - 1
    for idx in range(num_outliers + 1):
        start = outliers[idx]
        stop = outliers[idx + 1]
        if start < 109805 < stop:
            print("stop here")
        inside_cond = np.logical_and(rem_args > start, rem_args < stop)
        inside_args = np.argwhere(inside_cond).squeeze()
        num_mins = np.count_nonzero(inside_cond)
        if num_mins >= 3:
            # this is where we can start to feel pretty confident.
            # But we'll still rotate each peak based on the average within the peak.
            # outside the peak, I guess just rotate based on the closest peak?
            first_min = np.min(inside_args)
            last_min = np.max(inside_args - (1 if num_mins % 2 == 0 else 0))
            diffs = arg_diffs[first_min:last_min]
            hghts = mid_heights[first_min:last_min]
            if num_mins > 4:
                diff_scores = (
                    (np.mean(diffs) - diffs) / np.std(diffs).clip(1e-5)
                ).clip(-3.0, 3.0)
                height_scores = (
                    (hghts - np.mean(hghts)) / np.std(hghts).clip(1e-5)
                ).clip(-3.0, 3.0)
                parity = (
                    np.mean(diff_scores[::2] + height_scores[::2])
                    - np.mean(diff_scores[1::2] + height_scores[1::2])
                ) < 0  # zero if even are the peaks, else odd are
            else:
                seg = normed[rem_args[first_min] : rem_args[last_min]]
                frac_re = np.sum(np.square(np.real(seg) / np.abs(seg)))
                if frac_re > 0.5:
                    nseg = np.real(seg) - np.mean(np.real(seg))
                    # frac_under = np.count_nonzero(
                    #     (np.real(seg) < nseg
                    # )) / len(seg)
                else:
                    nseg = np.imag(seg) - np.mean(np.imag(seg))
                    # frac_under = np.count_nonzero(
                    #     (np.imag(seg) < np.mean(np.imag(seg)))
                    # ) / len(seg)
                parity = (
                    np.mean(np.abs(nseg[: diffs[0]]))
                    - np.mean(np.abs(nseg[diffs[0] :]))
                    > 0
                )
            # next, average across each peak.
            rot_angles = np.zeros((num_mins - 1) // 2, dtype=np.complex128)
            for jdx in range((num_mins - 1) // 2):
                first = rem_args[inside_args[jdx * 2]]
                mid = rem_args[inside_args[jdx * 2 + 1]]
                last = rem_args[inside_args[jdx * 2 + 2]]
                parity_av = np.mean(normed[first:mid]) - np.mean(normed[mid:last])
                parity_av /= np.abs(parity_av)
                if parity:
                    parity_av *= -1.0
                # first_pk = rem_args[inside_args[jdx * 2 + parity]]
                # last_pk = rem_args[inside_args[jdx * 2 + 1 + parity]]
                # wav_av = np.mean(nfold[first:last])
                # peak_av = np.mean(normed[first_pk:last_pk])
                rot_angles[jdx] = np.conj(parity_av)
                # rot_angles[jdx] = np.conj(
                #     wav_av * np.sign(np.real(wav_av / peak_av)) / np.abs(wav_av)
                # )
                rotated[first:last] = normed[first:last] * rot_angles[jdx]
                if first < 109805 < last:
                    print(first, last, rot_angles[jdx], wav_av, parity_av)
            # finally, do the parts on either side of the peaks...
            rotated[start : rem_args[first_min]] = (
                normed[start : rem_args[first_min]] * rot_angles[0]
            )
            rotated[rem_args[last_min] : stop] = (
                normed[rem_args[last_min] : stop] * rot_angles[-1]
            )

        else:
            # hard to tell what's going on really. We have one peak or less.
            # get the distance-weighted average, and rotate by that.
            wav_av = np.mean(nfold[start:stop])
            nsq_av = np.mean(nsq[start:stop])
            rot_angle = np.conj(
                wav_av * np.sign(np.real(wav_av / nsq_av)) / np.abs(wav_av)
            )
            diff = rotated[start - 1] - normed[start] * rot_angle
            ndiff = rotated[start - 1] + normed[start] * rot_angle
            result = np.sign(np.abs(ndiff) - np.abs(diff))
            rotated[start:stop] = normed[start:stop] * rot_angle * result

    _, (ax1, ax2) = plt.subplots(2, 1, sharex="col")
    ax1.imshow(cols, aspect="auto")
    ax2.plot(np.real(asym_mean))
    ax2.plot(np.abs(rot_conv))
    ax2.plot(asym_trace)
    ax2.plot(np.real(normed))
    ax2.plot(np.imag(normed))
    ax2.plot(real_norm, linestyle=":")
    ax2.plot(imag_norm, linestyle=":")
    ax2.plot(ratio, linestyle="--")
    ax2.scatter(asym_pk_arg, -4.5 * np.ones_like(asym_pk_arg))
    ax2.plot(np.real(rotated))
    # ax2.legend(
    #     [
    #         "asymmean",
    #         "rotconv",
    #         "asymtrace",
    #         "re(normed)",
    #         "im(normed)",
    #         "realnorm",
    #         "imnorm",
    #         "ratio",
    #         "rotated",
    #         "crossing",
    #     ]
    # )
    plt.show()

    # _, ax = plt.subplots(1, 1)
    # ax.plot(real_norm)
    # ax.plot(imag_norm)
    # ax.plot(np.log2(ratio))
    # plt.show()
    return (real_norm * ratio, imag_norm / ratio, ratio, cwt_real, cwt_imag)


def flip_signal(signal):
    smoothed, normed = smooth_and_norm_real(signal, smoothing=20.0)

    r2, *_ = compute_r2(normed.clip(-3.0, 3.0), smoothed, winsize=255)

    curv = get_findiff_curvature(smoothed)

    bmp = 21 / 60  # max breaths per second
    fs = 5  # sample rate
    max_curv_exag = (2 * np.pi * bmp / fs) ** 4

    curv_exag = (curv * np.abs(curv)) / max_curv_exag

    curv_exag_sm = correlate(curv_exag, np.ones(1001) / 10.0, mode="same")

    past_3 = np.argwhere(np.abs(normed) > 5.0).squeeze()
    num_outliers = len(past_3)
    outliers = np.zeros(num_outliers + 2, dtype=int)
    outliers[1:-1] = past_3
    outliers[-1] = len(signal) - 1

    righted = np.zeros_like(signal)
    stdev = 20.0
    windows = np.zeros_like(signal)

    for idx in range(num_outliers + 1):
        start = outliers[idx]
        stop = outliers[idx + 1]

        win_range = np.minimum(np.arange(stop - start), np.arange(stop - start)[::-1])
        window = 1.0 - np.exp(-0.5 * np.square(win_range.clip(max=3 * stdev) / stdev))
        windows[start:stop] = window

        right = np.sum(curv_exag[start:stop] * window) <= 0.0

        if right:
            righted[start:stop] = normed[start:stop]
        else:
            righted[start:stop] = -normed[start:stop]

    # _, ax = plt.subplots(1, 1)
    # ax.plot(normed)
    # ax.plot(smoothed)
    # ax.plot(windows)
    # ax.plot(righted, linestyle="--")
    # ax.plot(curv_exag)
    # ax.plot(curv_exag_sm)
    # ax.plot(r2)
    # ax.legend(["norm", "sm", "win", "right", "curv", "curvsm", "r2"])
    # plt.show()

    return smoothed, normed, r2, righted, curv_exag


def flip_components_indiv_and_combine(sig_x, sig_y, segments):
    """Normalizes the traces, then compares them and weights the wavier one higher."""
    sig_x_copy = np.copy(sig_x)
    sig_y_copy = np.copy(sig_y)

    for sdx, seg in enumerate(segments[:-1]):
        scale_x = flip_signal(sig_x[seg:segments[sdx + 1]])
        scale_y = flip_signal(sig_y[seg:segments[sdx + 1]])
        sig_x_copy[seg:segments[sdx + 1]] *= scale_x
        sig_y_copy[seg:segments[sdx + 1]] *= scale_y

    omega = 20.0
    fs = 5.0
    freqs = np.logspace(0.1, -1.4, 150)  # ~50-85 are breathing frequencies
    widths_morlet = omega * fs / (freqs[55:80] * 2 * np.pi)
    real_wave = wavefinding_cwt(real_righted.clip(-3.0, 3.0), widths_morlet, omega)
    mags_real = np.sum(np.square(np.abs(real_wave)), axis=0)
    imag_wave = wavefinding_cwt(imag_righted.clip(-3.0, 3.0), widths_morlet, omega)
    mags_imag = np.sum(np.square(np.abs(imag_wave)), axis=0)

    rr_align = np.zeros_like(real_righted)
    ir_align = np.zeros_like(imag_righted)

    past_3 = np.argwhere(
        np.logical_or(np.abs(real_norm) > 5.0, np.abs(imag_norm) > 5.0)
    ).squeeze()
    num_outliers = len(past_3)
    outliers = np.zeros(num_outliers + 2, dtype=int)
    outliers[1:-1] = past_3
    outliers[-1] = len(real_righted) - 1

    stdev = 20.0

    for idx in range(num_outliers + 1):
        start = outliers[idx]
        stop = outliers[idx + 1]

        if start < 97250 < stop:
            print("shit")

        win_range = np.minimum(np.arange(stop - start), np.arange(stop - start)[::-1])
        window = 1.0 - np.exp(-0.5 * np.square(win_range.clip(max=3 * stdev) / stdev))

        agree = (
            np.sum(real_righted[start:stop] * imag_righted[start:stop] * window) >= 0.0
        )

        if agree:
            rr_align[start:stop] = real_righted[start:stop]
            ir_align[start:stop] = imag_righted[start:stop]
        else:
            # who has the higher sum of curvatures on their side?
            real_flip = np.sign(
                np.abs(np.sum(real_curv[start:stop]))
                - np.abs(np.sum(imag_curv[start:stop]))
            )
            rr_align[start:stop] = real_righted[start:stop] * real_flip
            ir_align[start:stop] = imag_righted[start:stop] * -real_flip

    real_frac = mags_real / (mags_real + mags_imag)
    imag_frac = 1.0 - real_frac

    result = real_frac * real_righted + imag_frac * imag_righted

    # _, ax = plt.subplots(1, 1)
    # ax.plot(real_righted)
    # # ax.plot(real_smooth)
    # ax.plot(imag_righted)
    # ax.plot(result)
    # # ax.plot(imag_smooth)
    # # ax.plot(result)
    # # ax.plot(real_nm_curv_conv + 3.0, linestyle="--")
    # # ax.plot(imag_nm_curv_conv + 3.0, linestyle="--")
    # # ax.plot(real_r2 - 3.0)
    # # ax.plot(imag_r2 - 3.0, linestyle="--")
    # # ax.plot(mags_real * 0.005)
    # # ax.plot(mags_imag * 0.005)
    # ax.plot(real_frac)
    # ax.plot(imag_frac)
    # ax.legend(["rr", "ir", "res", "rf", "if"])
    # plt.show()

    return result

def adjust_for_zero_trend(signal, lam=25.0):
    new_sig = np.copy(signal)
    sm = cspline1d(signal, lam)
    segs = np.argwhere(np.logical_and(sm[1:] * sm[:-1] < 0.0, sm[1:] < sm[:-1])).squeeze()
    segs[0] = 0
    for idx in range(len(segs) - 1):
        start = segs[idx]
        stop = segs[idx + 1]
        seg_offset = np.mean(signal[start:stop])
        new_sig[start:stop] -= seg_offset
        bnd_offset = np.mean(np.cumsum(new_sig[start:stop]))
        new_sig[start] -= 0.5 * bnd_offset
        new_sig[stop] += 0.5 * bnd_offset
    
    return new_sig, sm

def find_periodic(signal, freqs):
    m = 3.0
    tl = signal.size
    fs = 5
    r2 = np.zeros((freqs.size, tl))
    lgs = np.around(m * fs / freqs)
    sss = np.zeros((freqs.size, tl))
    sqsig = np.square(signal)
    for fdx, freq in enumerate(freqs):
        lg = round(m * fs / freq)
        cos = np.cos(2.0 * m * np.pi * np.arange(lg) / lg)
        sin = np.sin(2.0 * m * np.pi * np.arange(lg) / lg)
        dcos = np.cos(4.0 * m * np.pi * np.arange(lg) / lg)
        dsin = np.sin(4.0 * m * np.pi * np.arange(lg) / lg)
        a = np.square(np.correlate(signal, cos) * 2.0 / lg)
        b = np.square(np.correlate(signal, sin) * 2.0 / lg)
        c = np.square(np.correlate(signal, dcos) * 2.0 / lg)
        d = np.square(np.correlate(signal, dsin) * 2.0 / lg)
        sss[fdx, :(tl - lg + 1)] = np.correlate(sqsig, np.ones(lg))  # sum of square signal
        part_sse = lg * 0.5 * (a + b + c + d)
        # sse = sss - part_sse, r2 = 1 - sse / sss -> simplified it
        # only works if we assume zero mean, which I do in the fit.
        r2[fdx, :(tl - lg + 1)] = part_sse / sss[fdx, :(tl - lg + 1)]
    best_freqs = np.argmax(correlate1d(r2, np.ones(250), mode="nearest"), axis=0)
    final_r2 = np.zeros((tl, ))
    final_lg = np.zeros((tl, ))
    for idx, fdx in enumerate(best_freqs):
        offset = round(lgs[fdx] / m)
        lg = int(lgs[fdx])
        final_lg[idx] = lg
        if idx + offset + lg >= tl:
            break
        sse = np.sum(np.square(signal[idx:idx + lg] - signal[idx + offset:idx + lg + offset]))
        cr2 = 1.0 - sse / max(sss[fdx, idx], sss[fdx, idx + offset])
        final_r2[idx:idx + offset + lg] = np.maximum(final_r2[idx:idx + offset + lg], cr2)
    _, ax = plt.subplots(1, 1)
    ax.imshow(correlate1d(r2, np.ones(250), mode="nearest"), aspect="auto")
    ax.plot(best_freqs, c="r")
    plt.show()
    return final_r2.clip(min=0.0), final_lg

# ricky = ricker(101, 10)
# morry = morlet2(101, 12, 1.7) * 2743 / 2168
# _, ax = plt.subplots(1, 1)
# ax.plot(ricky, c="r")
# ax.plot(morry, c="k")
# plt.show()

# %%
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

rms0 = get_rms(n[:, 0], winsize=65)
rms1 = get_rms(n[:, 1], winsize=65)
rms_comb = np.sqrt(np.square(rms0) + np.square(rms1))

sig0, sm0 = adjust_for_zero_trend(n[:, 0])
sig1, sm1 = adjust_for_zero_trend(n[:, 1])

omega = 10.0
fs = 5.0
freqs = np.logspace(0.1, -1.4, 150)  # indices ~50-85 are breathing frequencies
widths_morlet = omega * fs / (freqs * 2 * np.pi)
z_wave = wavefinding_cwt(np.cumsum(sig0 + 1.0j * sig1), widths_morlet, omega)
mags_z = np.abs(z_wave) / rms_comb.clip(1e-5)

_, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
# ax1.plot(sig0)
# ax1.plot(sig1)
ax1.plot(n[:, 0])
ax1.plot(n[:, 1])
ax1.plot(rms0, linestyle=":")
ax1.plot(rms1, linestyle=":")
# ax1.plot(sm0, linestyle=":")
# ax1.plot(sm1, linestyle=":")
ax2.plot(np.cumsum(sig0))
ax2.plot(np.cumsum(sig1))
ax3.imshow(mags_z.clip(max=np.percentile(mags_z, 98.0)), aspect="auto")
plt.show()

r2_x, lgs = find_periodic(n[:, 0], freqs[50:80])
r2_y, _ = find_periodic(n[:, 1], freqs[50:80])

x_env = np.abs(hilbert(n[:, 0]))
y_env = np.abs(hilbert(n[:, 1]))

r2 = np.maximum(r2_x, r2_y)
count = np.sum(r2 >= 0.5)
rmses = np.sqrt(np.sum((r2 >= 0.5).astype(float) * (np.square(n[:, 0]) + np.square(n[:, 1]))) / count)
condit = np.logical_and(r2 >= 0.5, np.logical_and(n[:, 0] < rmses * 10.0, n[:, 1] < rmses * 10.0))
starts = np.argwhere(np.logical_and(condit[1:], np.logical_not(condit[:-1]))).squeeze()
stops = np.argwhere(np.logical_and(np.logical_not(condit[1:]), condit[:-1])).squeeze()

mx_lead, mx_lag, bx_lead, bx_lag, rmsx_lead, rmsx_lag = fit_linear_lead_and_lag(n[:, 2], winsize=155)
my_lead, my_lag, by_lead, by_lag, rmsy_lead, rmsy_lag = fit_linear_lead_and_lag(n[:, 3], winsize=155)

x_rms_frac = rmsx_lead / (rmsx_lead + rmsx_lag)
y_rms_frac = rmsy_lead / (rmsy_lead + rmsy_lag)
bx_best = x_rms_frac * bx_lag + (1.0 - x_rms_frac) * bx_lead # np.where(rmsx_lead < rmsx_lag, bx_lead, bx_lag)
by_best = y_rms_frac * by_lag + (1.0 - y_rms_frac) * by_lead # np.where(rmsy_lead < rmsy_lag, by_lead, by_lag)

b_best = np.stack((bx_best, by_best), axis=-1)

res_x = rpt.KernelCPD(kernel="linear", min_size=100).fit(n[:, 2:4]).predict(pen=150)
# res_y = rpt.KernelCPD(kernel="linear", min_size=100).fit(n[:, 4:6]).predict(pen=2500)

_, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(timestamps, my_lead)
ax1.plot(timestamps, my_lag)
ax1.plot(timestamps, bx_best)
ax1.plot(timestamps, by_best)
ax1.plot(timestamps, rmsy_lead)
ax1.plot(timestamps, rmsy_lag)
ax1.plot(timestamps, n[:, 2:4], alpha=0.6)
ax1.plot(timestamps, n[:, 4:6], alpha=0.2)
ax2.plot(timestamps, n[:, 0:2])
ax2.axhline(y=rmses * 10.0)
ax2.axhline(y=rmses * -10.0)
ax2.plot(timestamps, r2 * 1e-3, c="k", alpha=0.1, linestyle=":")
ax2.plot(timestamps, x_env)
ax2.plot(timestamps, y_env)

for idx, ln in enumerate(res_x[:-1]):
    ax1.axvline(x=timestamps[ln])
    frac = np.count_nonzero(condit[ln:res_x[idx + 1]]) / (res_x[idx + 1] - ln)
    if frac < 0.5:
        ax1.axvspan(timestamps[ln], timestamps[res_x[idx + 1]], color="r", alpha=0.2, ec=None)
for idx in range(starts.size):
    ax2.axvspan(timestamps[starts[idx]], timestamps[stops[idx]], color="g", alpha=0.2)
# for ln in res_y[:-1]:
#     ax1.axvline(x=timestamps[ln], c="r")

plt.show()

final = flip_components_indiv_and_combine(n[:, :2], res_x)
final_sm = cspline1d(final, lamb=5.0)

pks, *_ = find_peaks(final_sm, prominence=0.7)
pk_dist = pks[1:] - pks[:-1]
apneas = pks[:-1][pk_dist > 30]

lgth = final.shape[0]
omega = 10.0
fs = 5.0
freqs = np.logspace(0.1, -1.4, 150)
widths_morlet = omega * fs / (freqs * 2 * np.pi)
cwt_morlet = wavefinding_cwt(final, widths_morlet, omega)
angles = np.angle(cwt_morlet)
mags = np.square(np.abs(cwt_morlet))
cols = np.mod(angles * 0.5 / np.pi + 0.5, 1)
col_spec = cols - 0.5
dcol = col_spec[:, 1:] - col_spec[:, :-1]
dcol[dcol < -0.5] += 1.0
dcol[dcol > 0.5] -= 1.0
dcols = cm.hsv(dcol.clip(-0.00625, 0.00625) * 80.0 + 0.5)[..., :3]
dcols *= (mags[:, 1:, None] / np.percentile(mags, 95.0)).clip(max=1.0)
cols = cm.hsv(cols)[..., :3]
cols *= mags[..., None]

# ddcol = dcol[50:85] - dcol[49:84]
scale = 500.0
ddcol = dcol[1:] - dcol[:-1]
rate_data = mags[50:85, 1:] * ddcol[49:84].clip(0.0) * np.exp(-scale * np.abs(dcol[50:85]))
inst_rate = np.sum(np.arange(50, 85)[:, None] * rate_data, axis=0) / np.sum(rate_data, axis=0).clip(1e-5)
inst_var = np.sum(np.square(np.arange(50, 85)[:, None] - inst_rate) * rate_data, axis=0) / np.sum(rate_data, axis=0).clip(1e-5)
inst_std = np.sqrt(inst_var)
inst_std_sm = correlate(inst_std, np.ones(31), mode="same")

_, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
ax1.imshow((ddcol * scale).clip(0.0, 1.0), aspect="auto")
ax2.imshow(np.exp(-scale * np.abs(dcol)), aspect="auto")
ax3.imshow(mags.clip(max=np.percentile(mags, 98.0)), aspect="auto")
ax4.imshow(rate_data, aspect="auto")
ax4.plot(inst_rate - 50.0, c="r", linestyle=":")
# ax4.plot(inst_std_sm, c="y", linestyle=":")
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
ax1.imshow(mags.clip(max=np.percentile(mags, 95.0)), aspect="auto")
ax2.plot(final_sm)
ax2.scatter(apneas, final_sm[apneas], c="k", marker="+", zorder=1000)
ax3.imshow(dcols, aspect="auto")
plt.show()

_, ax = plt.subplots(1, 1)
ax.hist(pk_dist * 0.2, bins=120, range=(0.1, 24.1))
plt.show()

_, ax = plt.subplots(1, 1, sharex=True)
ax.plot(timestamps, final)
plt.show()

quit()

n1 = norm_complex(n0_resz, n1_resz)
n2 = norm_complex(n2_resz, n3_resz)

n1 = smooth_and_norm_complex_stitch(n0_resz, n1_resz, n2_resz, n3_resz)

m1, r1, res1, skew1 = compute_stats(n[:, 0], n[:, 1], winsize)
m2, r2, res2, skew2 = compute_stats(n[:, 2], n[:, 3], winsize)

# sig1 = smooth_and_norm(res1)
# sig2 = smooth_and_norm(res2)


# %%
# CWT

lgth = n1.shape[0]
omega = 2.0
fs = 5.0
freqs = np.logspace(0.1, -1.4, 150)
widths_morlet = omega * fs / (freqs * 2 * np.pi)
cwt_morlet = wavefinding_cwt(n1, widths_morlet, omega)
angles = np.angle(cwt_morlet)
mags = np.square(np.abs(cwt_morlet))
cols = np.mod(angles * 0.5 / np.pi + 0.5, 1)
col_spec = cols - 0.5
dcol = col_spec[:, 1:] - col_spec[:, :-1]
dcol[dcol < -0.5] += 1.0
dcol[dcol > 0.5] -= 1.0
dcols = cm.hsv(dcol.clip(-0.0125, 0.0125) * 40.0 + 0.5)[..., :3]
# dcols = cm.twilight_shifted(dcol.clip(-0.05, 0.05) * 10.0 + 0.5)[..., :3]
dcols *= (mags[:, 1:, None] / np.percentile(mags, 95.0)).clip(max=1.0)
cols = cm.hsv(cols)[..., :3]
cols *= mags[..., None]

sharps = np.sum(mags[:50], axis=0)
sharps *= 0.001  # empirical constant
sharps[0] = 0.0
sharps[-1] = 0.0

fatties = np.sum(mags[-50:], axis=0)
fatties *= 0.002
fatties[0] = 0.0
fatties[-1] = 0.0

mainline = np.sum(mags[50:-50], axis=0)
mainline *= 0.004
mainline[0] = 1.0
mainline[-1] = 1.0

summed = np.sum(mags, axis=0, keepdims=True)
wt_mean = np.sum(np.arange(freqs.size)[:, None] * mags, axis=0, keepdims=True) / summed
stdev = np.sqrt(
    np.sum(mags * np.square(np.arange(freqs.size)[:, None] - wt_mean), axis=0) / summed
)

movement = np.logical_or(sharps > 0.5, np.abs(n1) > 5.00).astype(int)
stationary = (mainline < 0.5).astype(int)

mvmt_start = np.argwhere(movement[1:] - movement[:-1] == 1).squeeze()
mvmt_end = np.argwhere(movement[1:] - movement[:-1] == -1).squeeze()
stn_start = np.concatenate(
    (
        np.array([-2]),
        np.argwhere(stationary[1:] - stationary[:-1] == 1).squeeze(),
        np.array([lgth]),
    )
)
stn_end = np.concatenate(
    (
        np.array([-1]),
        np.argwhere(stationary[1:] - stationary[:-1] == -1).squeeze(),
        np.array([lgth + 1]),
    )
)

before_args = np.argmax(
    np.mod(stn_end[:, None] - mvmt_start[None, :], lgth * 2), axis=0
)
before_okay = np.logical_and(
    0 < mvmt_start - stn_end[before_args], mvmt_start - stn_end[before_args] < 15
)
before_dist = (mvmt_start - stn_start[before_args]) * before_okay.astype(int)
mvmt_start_adj = mvmt_start - before_dist

after_args = np.argmax(np.mod(mvmt_end[None, :] - stn_start[:, None], lgth * 2), axis=0)
after_okay = np.logical_and(
    0 < stn_start[after_args] - mvmt_end, stn_start[after_args] - mvmt_end < 15
)
after_dist = (stn_end[after_args] - mvmt_end) * after_okay.astype(int)
mvmt_end_adj = mvmt_end + after_dist

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
ax1.imshow(mags.clip(max=np.percentile(mags, 95.0)), aspect="auto")
ax2.plot(np.real(n1))
ax2.plot(np.imag(n1))
ax2.plot(np.abs(n1))
ax2.plot(sharps.clip(max=2.0) * 10.0, linestyle=":")
# ax2.plot(mainline, c="k")
# ax2.plot(stdev[0], c="b")
# ax2.plot(movement * 3.0, c="k")
# ax2.plot(stationary * 2.0, c="b")
ax2.scatter(mvmt_start_adj, np.ones_like(mvmt_start_adj), c="k", zorder=1000)
ax2.scatter(mvmt_end_adj, np.ones_like(mvmt_end_adj), c="b", zorder=1001)
ax3.imshow(dcols, aspect="auto")
plt.show()

fig, ax = plt.subplots(1, 1)
ax.imshow(dcols, aspect="auto")
fig.set_size_inches(10, 4)
plt.show()


# %%
# dto = datetime.strptime(dt, "%y%m%d_%H%M%S")
# start = dto.hour + dto.minute / 60.0 + dto.second / 3600.0
# end = start + n.shape[0] / (5.0 * 3600.0)
# t = start + np.arange(n.shape[0]) / (5.0 * 3600.0) - (24.0 if start > 12 else 0.0)

# nfft = 64
# f, tfft, s = spectrogram(sig1, fs=5.0, nfft=nfft, mode="magnitude", nperseg=nfft)
# spec = 10.0 * np.log10(np.square(s))

# %%
# _, (ax1, ax2) = plt.subplots(2, 1)
# ax1.imshow(
#     np.flip(spec, axis=0),
#     vmin=np.percentile(spec, 20.0),
#     vmax=np.percentile(spec, 99.5),
#     extent=[t[0], t[-1], f[0], f[-1]],
# )
# ax2.plot(m1)
# ax2.plot(r2 * 2.0 + 3)
# plt.show()


# %%

omega = 10.0
widths_morlet = omega * fs / (freqs * 2 * np.pi)
cwt_morlet = wavefinding_cwt(n1, widths_morlet, omega)
angles = np.angle(cwt_morlet)
mags = np.abs(cwt_morlet)
cols = np.mod(angles * 0.5 / np.pi + 0.5, 1)
col_spec = cols - 0.5
dcol = col_spec[:, 1:] - col_spec[:, :-1]
dcol[dcol < -0.5] += 1.0
dcol[dcol > 0.5] -= 1.0
dcols = cm.hsv(dcol.clip(-0.0125, 0.0125) * 40.0 + 0.5)[..., :3]
# dcols = cm.twilight_shifted(dcol.clip(-0.05, 0.05) * 10.0 + 0.5)[..., :3]
dcols *= (mags[:, 1:, None] / np.percentile(mags, 95.0)).clip(max=1.0)
cols = cm.hsv(cols)[..., :3]
cols *= mags[..., None]

sharps = np.sum(mags[:50], axis=0)
sharps *= 0.001 if omega > 4.0 else 0.005

fatties = np.sum(mags[-50:], axis=0)
fatties *= 0.002 if omega > 4.0 else 0.01

mainline = np.sum(mags[50:-50], axis=0)
mainline *= 0.004 if omega > 4.0 else 0.01

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
ax1.imshow(mags.clip(max=np.percentile(mags, 95.0)), aspect="auto")
ax2.plot(np.real(n1))
ax2.plot(np.imag(n1))
ax2.plot(sharps)
ax2.plot(fatties)
ax2.plot(mainline, c="k")
ax3.imshow(dcols, aspect="auto")
fig.set_size_inches(10, 10)
plt.show()

# %%
# _, ax = plt.subplots(1, 1)
# ax.plot(timestamps, n[:, :2] * 0.2 / np.std(n))
# ax.plot(timestamps[:-60], n[60:, 2:] * 0.2 / np.std(n))
# # ax.plot(timestamps[winsize - 1 :-1], np.abs((m1[1:] - m1[:-1]) / np.square(m1[1:] + m1[:-1])), c="k")
# # ax.plot(timestamps[: 1 - winsize - 60 - 1], np.abs((m2[61:] - m2[60:-1]) / np.square(m2[61:] + m2[60:-1])), c=(0.4, 0.4, 0.4))
# ax.plot(timestamps[winsize - 1 :], medfilt(m1, 21), c="k")
# ax.plot(timestamps[: 1 - winsize - 60], medfilt(m2[60:], 21), c=(0.4, 0.4, 0.4))
# plt.show()


quit()

widths_morlet = np.arange(7, 40)
widths_ricker = np.arange(14, 80) * 0.2
cwt_morlet = cwt(sig_alter, morlet_real, widths_morlet)
cwt_morlet /= np.max(np.abs(cwt_morlet))
cwt_morlet[0] = 0.0
cwt_ricker = cwt(sig_alter, ricker, widths_ricker)
cwt_ricker /= np.max(np.abs(cwt_ricker))
cwt_ricker[0] = 0.0
# fig, ax = plt.subplots(1, 1)
# ax.imshow(np.concatenate((cwt_morlet, cwt_ricker), axis=0), aspect="auto")
# fig.set_size_inches(10, 5)
# plt.show()

morlet_slopes = get_slopes(cwt_morlet)
ricker_slopes = get_slopes(cwt_ricker)

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
# ax1.imshow(np.sum(morlet_slopes, axis=0), aspect="auto")
# ax2.imshow(np.sum(np.sign(morlet_slopes), axis=0), aspect="auto")
# ax3.imshow(np.sum(ricker_slopes, axis=0), aspect="auto")
# ax4.imshow(np.sum(np.sign(ricker_slopes), axis=0), aspect="auto")
# fig.set_size_inches(10, 10)
# plt.show()

morlet_pve_cond = np.sum(np.sign(morlet_slopes), axis=0) == 8.0
morlet_nve_cond = np.sum(np.sign(morlet_slopes), axis=0) == -8.0
morlet_crop = np.copy(cwt_morlet[1:-1, 1:-1])
morlet_crop[morlet_pve_cond] = 4.0
morlet_crop[morlet_nve_cond] = 3.0

ricker_pve_cond = np.sum(np.sign(ricker_slopes), axis=0) == 8.0
ricker_nve_cond = np.sum(np.sign(ricker_slopes), axis=0) == -8.0
ricker_crop = np.copy(cwt_ricker[1:-1, 1:-1])
ricker_crop[ricker_pve_cond] = 4.0
ricker_crop[ricker_nve_cond] = 3.0

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.imshow(morlet_crop, aspect="auto")
# ax2.imshow(ricker_crop, aspect="auto")
# fig.set_size_inches(10, 5)
# plt.show()

cwsum = np.sum(cwt_ricker, axis=0)
print(cwsum.shape, sig_alter.shape)
reszf = np.sum(np.abs(cwsum)) / np.sum(np.abs(sig_alter))

# _, ax = plt.subplots(1, 1)
# ax.plot(np.sum(cwt_morlet, axis=0))
# ax.plot(np.sum(cwt_ricker, axis=0))
# ax.plot(sig_alter * reszf)
# plt.show()


# %%
# import numpy as np
# import matplotlib.pyplot as plt
# a = np.linspace(0.0, 4.0, 41)
# a2 = a + 0.5 * np.sin(a) + 0.5 * np.cos(a * 0.1)
# b = np.mod(a2, 1.0)
# c = np.mod(b[1:] - b[:-1], 1.0)
# print(c)
# _, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(a + 0.5 * np.sin(a))
# ax2.plot(c)
# %%

# dx = cwt_morlet[1:-1, 1:] - cwt_morlet[1:-1, :-1]
# pdx = dx[:, 1:] * dx[:, :-1]
# tpx = pdx < 0.0

# dy = cwt_morlet[1:, 1:-1] - cwt_morlet[:-1, 1:-1]
# pdy = dy[1:] * dy[:-1]
# tpy = pdy < 0.0

# px = np.copy(cwt_morlet)[1:-1, 1:-1]
# px[tpx] = -4

# py = np.copy(cwt_morlet)[1:-1, 1:-1]
# py[tpy] = -4

# dxy = cwt_morlet[1:, 1:] - cwt_morlet[:-1, :-1]
# pdxy = dxy[1:, 1:] * dxy[:-1, :-1]
# tpxy = pdxy < 0.0
# pxy = np.copy(cwt_morlet)[1:-1, 1:-1]
# pxy[tpxy] = -4

# dyx = cwt_morlet[:-1, 1:] - cwt_morlet[1:, :-1]
# pdyx = dyx[:-1, 1:] * dyx[1:, :-1]
# tpyx = pdyx < 0.0
# pyx = np.copy(cwt_morlet)[1:-1, 1:-1]
# pyx[tpyx] = -4

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
# ax1.imshow(px, aspect="auto")
# ax2.imshow(py, aspect="auto")
# ax3.imshow(pxy, aspect="auto")
# ax4.imshow(pyx, aspect="auto")
# fig.set_size_inches(10, 5)
# plt.show()

# the logical and condition finds all pixels where the pixel is either above or
# below all of its neighbours - ie the relative extrema
# the resultant points have peak width (y coordinate)
# as well as timing information (x coordinate)
# this can be used to
# pall = np.copy(cwt_morlet)[1:-1, 1:-1]
# statn_cond = np.logical_and(np.logical_and(tpx, tpy), np.logical_and(tpxy, tpyx))
# slope_cond = np.logical_and(
#     dx[:, :-1] * dy[:-1] > 0.0,
#     np.logical_and(dy[:-1] * dxy[:-1, :-1] > 0.0, dy[:-1] * dyx[:-1, :-1] > 0.0),
# )
# pall[np.logical_and(np.logical_and(tpx, tpy), np.logical_and(tpxy, tpyx))] = -4
# pall[np.logical_and(slope_cond, slope_cond)] = 4.0
# cwsum = np.sum(cwt_morlet, axis=0)
# print(cwsum.shape, sig_alter.shape)
# reszf = np.sum(np.abs(cwsum)) / np.sum(np.abs(sig_alter))


# get the r^2 value for that fit.
# r2, err, sumer, sumvar = compute_r2(sig_alter * reszf, cwsum, winsize=127)

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.imshow(pall, aspect="auto")
# ax2.plot(cwsum)
# ax2.plot(sig_alter * reszf)
# ax2.plot(err)
# ax2.plot(r2 * 50.0)
# ax2.plot(sumer)
# ax2.plot(sumvar)
# fig.set_size_inches(10, 5)
# plt.show()


# %%
# data_array part for segmentation. Slope might already be good enough for this.
# this is a long calculation...
winsize = 60
sig_len = data_array.shape[0]
zsq = np.zeros(sig_len - winsize)
for idx in trange(sig_len - winsize):
    win = data_array[idx : idx + winsize]
    avg = np.mean(win, axis=0)
    std = np.std(win, axis=0).clip(1e-5)
    normed = (win - avg) / std
    curr = (data_array[idx + winsize] - avg) / std
    zsq[idx] = np.min(
        linalg.norm(
            np.reshape(normed - curr[None, ...], (winsize, 12 * 20 * 2)), axis=1
        )
    )

# %%
# Plot zsq
fig, ax = plt.subplots(1, 1)
ax.plot(np.log10(zsq).clip(max=5) / 5)  # , marker=".", linestyle="None")
ax.plot(n[winsize + 1 :] / np.max(n))
# ax.set_ylim([0, 1e5])
fig.set_size_inches(10, 5)
plt.show()


# %%
fig, ax = plt.subplots(1, 1)
ax.plot(timestamps, sig1)
ax.plot(timestamps[:-60], sig2)
fig.autofmt_xdate()
plt.show()

# %%
_, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(timestamps, res1)
ax1.plot(timestamps[:-60], res2)
ax1.plot(timestamps, np.sign(skew1))
ax1.plot(timestamps[:-60], np.sign(skew2))
ax2.plot(timestamps[hw:-hw], m1)
ax2.plot(timestamps[hw : -60 - hw], m2)
ax2.plot(timestamps[hw:-hw], r1 * 2.0 + 3.0)
ax2.plot(timestamps[hw : -60 - hw], r2 * 2.0 - 5.0)
ax2.plot(timestamps, np.sign(skew1) + 5.1)
plt.show()


# %%+
start = 50000
lgth = 1000
sig_alter = np.copy(sig1[start : start + lgth])
sig_alter[112:128] = -1.5
res, comp, seg = correlate_peaks(sig_alter, 6, 240, 5, 30)
_, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(seg)
ax1.plot(comp, linestyle=":")
ax2.plot(res)
plt.show()


# %%

x = np.linspace(-2, 2, 100)
m = morlet_real(800, 80)
_, ax = plt.subplots(1, 1)
ax.plot(m)
plt.show()

# %%
# Pick the maximum absolute value point, start from there, work forwards and backwards
