from datetime import datetime
import gzip
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import numpy.linalg as linalg
import pytz
from scipy.ndimage import correlate1d, maximum_filter1d
from scipy.stats import multivariate_normal as mvnd
from scipy.signal import (
    correlate,
    cspline1d,
    medfilt,
    morlet2,
)
from tqdm import trange, tqdm

from video_writer import VideoWriter


def cwt_mag_cmpr(sig_x, sig_y, widths, omega, cmpr=50, clipmag=5.0e-3):
    lg = int(np.ceil(len(sig_x) / cmpr))
    half_diff = (lg * cmpr - len(sig_x)) // 2
    output = np.zeros((len(widths), lg))
    sig_pad = np.zeros((lg * cmpr), dtype=np.complex128)
    sig_pad[half_diff : len(sig_x) + half_diff] = sig_x
    sig_pad[half_diff : len(sig_x) + half_diff] += 1.0j * sig_y
    full = np.zeros(lg * cmpr)
    for ind, width in enumerate(widths):
        # go for an odd window length about 8x the length of the width
        N = round(4 * width - 0.5) * 2 + 1
        N = np.min([N, lg * cmpr])
        wavelet = morlet2(N, width, omega)
        # using correlate instead of convolve
        full.resize(lg * cmpr)
        np.abs(correlate(sig_pad.astype(np.complex128), wavelet, mode="same"), out=full)
        np.clip(full, 0.0, clipmag, out=full)
        # compress that down in the time axis by a factor of cmpr
        full.resize((lg, cmpr))
        output[ind] = full.sum(axis=1)
    return output


def smooth(signal, sm_half):
    cs = np.cumsum(
        np.concatenate(
            (np.full(sm_half, signal[0]), signal, np.full(sm_half, signal[-1]))
        )
    )
    sm = (cs[2 * sm_half :] - cs[: -2 * sm_half]) / (2.0 * sm_half)


def categorize_aper(shape, r2_x, r2_y, ab_x, ab_y, pos_x, pos_y):
    sm = 81
    sm_pos_x = medfilt(pos_x, sm)
    sm_pos_y = medfilt(pos_y, sm)

    dx = np.abs(pos_x - sm_pos_x)
    dy = np.abs(pos_y - sm_pos_y)

    med_dx = np.median(dx)
    med_dy = np.median(dy)
    x_cond = dx > 12.0 * med_dx
    y_cond = dy > 12.0 * med_dy

    exp_k = 4
    either = np.logical_or(x_cond, y_cond).astype(np.float32)
    expand = np.concatenate(
        (np.full(exp_k, either[0]), either, np.full(exp_k, either[-1]))
    )
    exp_cs = np.cumsum(expand)
    mov = exp_cs[2 * exp_k :] - exp_cs[: -2 * exp_k] > 0.5
    mov_mult = 1.0 + 9.0 * medfilt(mov.astype(np.float32), 2 * exp_k + 1)

    best_r2 = np.maximum(r2_x, r2_y).clip(0.05, 0.99)
    sqmag = np.sum(np.square(ab_x), axis=1) + np.sum(np.square(ab_y), axis=1).clip(
        1e-12
    )
    log_mag = np.log(sqmag)
    avg_log_mag = np.mean(log_mag[best_r2 >= 0.8])
    std_log_mag = np.std(log_mag[best_r2 >= 0.8])

    oob_buf = 101
    mov_buf = 7

    # rate non-periodic by magnitude, and by med_r^2
    # if it's low r^2 and low magnitude, it's probably out of bed
    # odds that it's nonperiodic, where 0.8 = 1:1 odds
    # z-score for magnitudes, and odds that the signal is small
    log_mag_z = (avg_log_mag - log_mag) / std_log_mag
    med_zed = medfilt(log_mag_z, 501).clip(-1.6, 1.6)
    diff_z = med_zed - log_mag_z
    std_diff = np.std(diff_z[best_r2 >= 0.8])
    odds_spike = (0.667 * diff_z / std_diff - 1.0).clip(0.0) * 4.0 + np.where(
        np.logical_or(diff_z > 0.0, mov), 0.1, 0.0
    ).astype(np.float32)
    odds_small = (log_mag_z - 1.0).clip(0.0)

    odds_nonper_inst = 16.0 * np.square(1.0 - best_r2)
    odds_mov = odds_nonper_inst * mov_mult * odds_spike
    p_mov = odds_mov / (odds_mov + 1.0)
    is_mov = medfilt((p_mov > 0.5).astype(np.float32), mov_buf)

    odds_nonper = odds_nonper_inst
    odds_ratio = odds_nonper * odds_small

    p_oob = medfilt(odds_ratio / (odds_ratio + 1.0), oob_buf)

    mov = p_mov > 0.8
    meb_mov = p_mov > 0.55
    mov[0] = True
    mov[-1] = True
    starts = np.argwhere(np.logical_and(np.logical_not(mov[:-1]), mov[1:])).squeeze()
    stops = np.argwhere(np.logical_and(np.logical_not(mov[1:]), mov[:-1])).squeeze()
    n_segs = starts.size
    oob = np.zeros(shape, dtype=bool)
    # TODO: When movement detection improves, move this to follow that
    # right now this will identify too many things as movement because the segments are
    # not correct and include twitches, sighs, and Tessa's movements.
    for idx in range(n_segs - 1):
        inb = np.arange(starts[idx], stops[idx + 1])
        seg = np.arange(stops[idx], starts[idx])
        mov[seg] = np.all(meb_mov[seg])
        avg_oob = np.mean(p_oob[inb])
        if avg_oob > 0.8:
            oob[inb] = True
    mov[0] = True
    mov[-1] = True
    return mov, oob, odds_nonper_inst > 1.0


def fit_piecewise_contin(y_bef, t_bef, y_aft, t_aft, sig):
    ctr = 0.5 * (t_bef[-1] + t_aft[0])
    rad = t_aft[0] - ctr
    xs = np.concatenate((t_bef, t_aft)) - ctr
    ys = np.concatenate((y_bef, y_aft))
    ys -= np.mean(ys)
    X = np.stack(
        (
            np.minimum(xs + rad, 0.0),
            np.maximum(xs - rad, 0.0),
            np.minimum(rad, np.maximum(xs, -rad)),
        ),
        axis=-1,
    )
    X -= np.mean(X, axis=0)
    tf = np.array(
        [[1.0, 0, -1], [1, 1, 0], [0, -1, 1]], dtype=np.float32
    )  # middle one is arbitrary... suspiciously like the old one xD
    # how is this derived? If slope before = A, slope after = B, and slope between = C
    # also if everything is moved so that mean = 0
    # we get that the y distance is 2 * rad * (A - C) and 2 * rad * (C - B)
    # We need to transform the space so that we can integrate on those variables.
    # Take the inverse of this matrix, and apply it to X to transform the space before
    # fitting. Those variables need to be small, OR if they have the same sign then the
    # slope is intermediate. Turned out to be simpler than the other solution, and works
    # better as well.
    Xtf = np.dot(X, np.linalg.inv(tf))
    covtf = np.linalg.inv(np.dot(Xtf.T, Xtf))
    pinvtf = linalg.pinv(Xtf)
    coeffstf = np.dot(pinvtf, ys)

    # conditions 1 and 2 check for matching sign, as well as the vertical strip of
    # the "small" condition (less than 1/2 of the standard deviation of the signal)
    # conditions 3 and 4 check for the horizontal strip of the "small" condition.
    cond_1 = mvnd.cdf(
        np.array([0.25 * sig / rad, np.inf, -0.25 * sig / rad], dtype=np.float32),
        mean=coeffstf,
        cov=sig ** 2 * covtf,
    )
    cond_2 = mvnd.cdf(
        np.array([0.25 * sig / rad, np.inf, -0.25 * sig / rad], dtype=np.float32),
        mean=-coeffstf,
        cov=sig ** 2 * covtf,
    )
    cond_3 = mvnd.cdf(
        np.array([np.inf, np.inf, 0.25 * sig / rad], dtype=np.float32),
        mean=coeffstf,
        cov=sig ** 2 * covtf,
    )
    cond_4 = mvnd.cdf(
        np.array([np.inf, np.inf, -0.25 * sig / rad], dtype=np.float32),
        mean=coeffstf,
        cov=sig ** 2 * covtf,
    )

    return cond_1 + cond_2 + cond_3 - cond_4


def find_discont_piecewise(
    pos_x, pos_y, mov, min_length=100, max_length=300, buf=20, sm=81
):
    px = medfilt(pos_x, sm)
    py = medfilt(pos_y, sm)
    starts = np.argwhere(np.logical_and(np.logical_not(mov[:-1]), mov[1:])).squeeze()
    stops = np.argwhere(np.logical_and(np.logical_not(mov[1:]), mov[:-1])).squeeze()
    n_seg = starts.size
    pdx = 0
    ndx = 1
    shift = np.zeros((px.size, 2), dtype=np.float32)
    while ndx < n_seg:
        bend = starts[pdx] - buf
        astr = stops[ndx] + buf
        bl = starts[pdx] - 2 * buf - stops[pdx]
        al = starts[ndx] - 2 * buf - stops[ndx]
        if bl < min_length:
            pdx += 1
            ndx = pdx + 1
            continue
        if al < min_length:
            ndx += 1
            continue
        bef = np.arange(bend - min(max_length, bl), bend)
        aft = np.arange(astr, astr + min(max_length, al))
        dur = np.arange(bend, astr)

        prob_x = fit_piecewise_contin(px[bef], bef, px[aft], aft, 1.2)
        prob_y = fit_piecewise_contin(py[bef], bef, py[aft], aft, 1.2)
        print(pdx, ndx, stops[pdx], stops[ndx], prob_x, prob_y)
        shift[dur, 0] = 1.0 - prob_x
        shift[dur, 1] = 1.0 - prob_y
        pdx = ndx
        ndx += 1
    return shift


def find_periodic_fit_best(signal_x, signal_y, freqs):
    phase_freq_x, k_freq_x, r2_x, ab_x, r2_med_x = find_periodic(signal_x, freqs)
    phase_freq_y, k_freq_y, r2_y, ab_y, r2_med_y = find_periodic(signal_y, freqs)

    return (
        r2_x,
        r2_y,
        ab_x,
        ab_y,
        k_freq_x,
        k_freq_y,
    )


def find_periodic(signal, freqs):
    m = 3.0
    tl = signal.size
    fs = 5
    sqsig = np.square(signal)

    # doing this with weighted least squares - 2 wavelengths, but weighted by
    # exp(-0.5 * (2x/lg)^2)
    # this effectively shrinks the wings down (and the function to fit), making errors
    # there count for less. The weight matrix is less simple, but the math is not much
    # more complicated
    m = 2.0
    N = 50
    k_0 = 0.5 * m * fs / freqs[0]
    k_m1 = 0.5 * m * fs / freqs[-1]
    k_exact = np.linspace(k_0, k_m1, N, dtype=np.float32)
    w_ab = np.zeros((N, tl, 2), dtype=np.float32)
    ssws = np.zeros((N, tl), dtype=np.float32)
    swbfms = np.zeros((N, tl), dtype=np.float32)
    sqsws = np.zeros((N, tl), dtype=np.float32)
    swdp = np.zeros((N, tl), dtype=np.float32)
    w_r2 = np.zeros((N, tl), dtype=np.float32)
    for kdx in trange(N, leave=False):
        ke = k_exact[kdx]
        lge = 2.0 * ke + 1.0
        k = round(ke)
        lg = 2 * k + 1
        rng = np.arange(lg) - k
        cos = np.cos(2.0 * m * np.pi * rng / lge)
        sin = np.sin(2.0 * m * np.pi * rng / lge)
        dcos = np.cos(4.0 * m * np.pi * rng / lge)
        dsin = np.sin(4.0 * m * np.pi * rng / lge)
        exp = np.exp(-0.5 * np.square(3.0 * rng / ke))
        sqrt_exp = np.exp(-0.25 * np.square(3.0 * rng / ke))
        A_W = np.square(cos) * exp
        C_W = np.square(sin) * exp
        E_W = np.square(dcos) * exp
        F_W = np.square(dsin) * exp
        A = np.sum(A_W)
        B = np.sum(cos * dcos * exp)
        C = np.sum(C_W)
        D = np.sum(sin * dsin * exp)
        E = np.sum(E_W)
        F = np.sum(F_W)
        G = B ** 2 - A * E
        H = D ** 2 - C * F
        w = np.correlate(signal, cos * exp, "same")
        x = np.correlate(signal, sin * exp, "same")
        y = np.correlate(signal, dcos * exp, "same")
        z = np.correlate(signal, dsin * exp, "same")
        wxyz = np.stack([w, x, y, z], axis=1)
        # the solution, (X'T X')^-1 * X'T * y' - X'T X' is a 4x4 with a simple inverse:
        a = (B / G) * y - (E / G) * w
        b = (D / H) * z - (F / H) * x
        c = (B / G) * w - (A / G) * y
        d = (D / H) * x - (C / H) * z
        abcd = np.stack([a, b, c, d], axis=1)
        abmag = np.maximum(np.sum(np.square(abcd[:, :2]), axis=1), 1e-9)
        cdmag = np.sum(np.square(abcd[:, 2:]), axis=1)
        # ratio = np.maximum(cdmag / abmag * 100, 1.0)
        # TODO: try other nights, see if ratio is needed. It may be.
        # Alternatively can choose only the maxima which have a ratio less than ~ 1.5
        # sum of square weighted signal - (y')^2 = exp * (sig)^2
        ssws = np.correlate(sqsig, exp, "same")
        # sum of weighted best fit model, squared (X' b)^2 = exp * (cos...dsin)^2 * abcd^2
        swbfms = np.square(a) * A
        swbfms += np.square(b) * C
        swbfms += np.square(c) * E
        swbfms += np.square(d) * F
        # squared sum of whitened signal - see wiki article on weighted least squares in diagonal weighting case
        # this part is for adjusting the average - because of the weighting, can no longer assume mean = 0
        # cross term from (y' - avg(y'))^2 turns out to be -2 times the square of the average
        # -1/lg * (y * sqrt(exp))
        sqsws = np.square(np.correlate(signal, sqrt_exp / np.sqrt(lg), "same"))
        denom = ssws - sqsws
        # abcd[:, 2:] /= ratio[:, None]
        # sum of weighted dot product - cross term from (y' - X' b)^2
        # -2 * y' X' b = -2 * exp * sig * (cos...dsin) * abcd = -2 * wxyz * abcd
        swdp = np.sum(wxyz * abcd, axis=1)
        w_r2[kdx] = (2.0 * swdp - swbfms - sqsws) / denom
        w_ab[kdx] = abcd[:, :2]

    pbar = tqdm(total=8, leave=False)
    pbar.set_description("finding best k values   ")
    # empirically, this blur is about 6 breaths long
    nbl = 37 * 3
    wr2_blur = correlate1d(w_r2, np.ones(nbl, dtype=np.float32) / nbl, mode="nearest")
    k_best_spline = cspline1d(np.argmax(wr2_blur, axis=0).astype(np.float32), 1.0)
    k_best_sm = np.around(k_best_spline).astype(int).clip(0, N - 1)
    best_ks = k_best_sm[None, :, None]
    pbar.update(1)
    pbar.set_description("creating rotation matrix")
    # ab is how  well cos and sin fit, respectively. From this we can determine phase.
    ab = np.take_along_axis(w_ab, best_ks, axis=0).squeeze()
    # start constructing a rotation matrix to rotate points by the inverse of that phase
    invert = np.zeros((*ab.shape, 2), dtype=np.float32)
    invert[:, 0, :] = ab
    invert[:, 1, 0] = -ab[:, 1]
    invert[:, 1, 1] = ab[:, 0]
    invert /= linalg.norm(ab, axis=1).clip(min=1e-9)[:, None, None]
    # multiplying ab by invert with an offset of 1 gets the sin and cos components, but rotated
    pbar.update(1)
    pbar.set_description("calculating phase diffs ")
    rotated = np.matmul(invert[:-1], ab[1:, :, None]).squeeze()
    u_bnd = 2 * m * np.pi / (2 * k_0 + 1.0)
    l_bnd = 1.25 * m * np.pi / (2 * k_m1 + 1.0)
    # finally, grab the rotated sin and cos components and use arctan2 to get the phase difference
    # my math seems to have been wrong, so we need to multipy by -1
    phase_diffs = -np.arctan2(rotated[:, 1], rotated[:, 0])
    # phase_diffs = np.where(phase_diffs <= 0.0, phase_diffs + 2.0 * np.pi, phase_diffs)
    cumulative_phase = np.cumsum(phase_diffs.clip(l_bnd, u_bnd))

    pbar.update(1)
    pbar.set_description("estimating frequencies  ")
    mean_pd = np.mean(
        phase_diffs[np.logical_and(phase_diffs < u_bnd, phase_diffs > l_bnd)]
    )
    avg_phase_offset = round(
        4.0 * np.pi / mean_pd
    )  # how long would the mean phase difference take to sum to 2 wavelengths?

    # provide an estimate of the instantaneous frequencies
    est_inst_freqs = np.zeros(tl, dtype=np.float32)
    est_inst_freqs[:-1] = phase_diffs * (fs / (2.0 * np.pi))
    est_inst_freqs[-1] = est_inst_freqs[-2]

    # provide the second estimate of frequencies
    smooth_k_freqs = 0.5 * m * fs / (k_best_spline * (k_m1 - k_0) / N + k_0)

    pbar.update(1)
    pbar.set_description("dynamically warping time")
    # apply phase warp, find the error and then the r^2 value
    interp = np.interp(
        cumulative_phase - 2.0 * np.pi,
        cumulative_phase,
        np.arange(cumulative_phase.size),
    )
    x_max = int(np.max(interp))
    resamp = np.interp(np.arange(1, x_max), interp, signal[:-1])
    err = signal[1:x_max] - resamp
    sse = np.correlate(
        np.square(err),
        np.ones(avg_phase_offset, dtype=np.float32) / avg_phase_offset,
        mode="full",
    )
    sst = np.correlate(
        sqsig[1:x_max],
        np.ones(avg_phase_offset, dtype=np.float32) / avg_phase_offset,
        mode="full",
    )
    r2_fast = 1.0 - sse / sst

    # go the other way, interpolate from the phase warp to the original
    pbar.update(1)
    pbar.set_description("reversing the warp      ")
    rev_interp = np.interp(interp, np.arange(tl), signal)

    rev_err = signal[:-1] - rev_interp
    rev_sse = np.correlate(
        np.square(rev_err),
        np.ones(avg_phase_offset, dtype=np.float32) / avg_phase_offset,
        mode="full",
    )
    rev_sst = np.correlate(
        sqsig[:-1],
        np.ones(avg_phase_offset, dtype=np.float32) / avg_phase_offset,
        mode="full",
    )
    rev_r2_fast = 1.0 - rev_sse / rev_sst

    gsn = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0], dtype=np.float32)
    gsn /= np.sum(gsn)

    pbar.update(1)
    pbar.set_description("calculating r^2         ")
    r2_max_fast = np.maximum(
        r2_fast[: 1 - avg_phase_offset], r2_fast[avg_phase_offset - 1 :]
    )
    rev_r2_max_fast = np.maximum(
        rev_r2_fast[: 1 - avg_phase_offset], rev_r2_fast[avg_phase_offset - 1 :]
    )

    pbar.update(1)
    pbar.set_description("smoothing r^2           ")
    fwd_r2 = np.correlate(
        np.concatenate(
            (
                np.ones(4, dtype=np.float32) * r2_max_fast[0],
                r2_max_fast,
                np.ones(tl - x_max + 3, dtype=np.float32) * r2_max_fast[-1],
            )
        ),
        gsn,
        mode="same",
    ).clip(0.0)[3:-3]
    rev_r2 = np.correlate(
        np.concatenate(
            (
                np.ones(3, dtype=np.float32) * rev_r2_max_fast[0],
                rev_r2_max_fast,
                np.ones(4, dtype=np.float32) * rev_r2_max_fast[-1],
            )
        ),
        gsn,
        mode="same",
    ).clip(0.0)[3:-3]

    pbar.update(1)
    pbar.set_description("medfilt of best r^2     ")
    skew_win = 8 * avg_phase_offset + 1

    # median filter - if the median is less than the segments marked periodic, it's
    # probably the middle of a long period of movement or a period of being out of bed.
    # Anything less than 1 minute is likely to be filtered out by the median filter.
    # This can be changed.
    periodic_r2 = np.maximum(fwd_r2, rev_r2)
    periodic_r2[0] = 0.0
    periodic_r2[-1] = 0.0
    r2_med = medfilt(periodic_r2, skew_win)

    pbar.update(1)
    pbar.set_description("Done.                   ")
    pbar.close()

    return est_inst_freqs, smooth_k_freqs, periodic_r2, ab, r2_med


def mag_arg_to_rgb_float(comp):
    hv = comp.astype(np.float32) / 255.0
    hv[..., 0] *= 4.0
    hsv = np.stack(
        (
            hv[..., 1],
            np.ones(comp.shape[:-1], dtype=np.float32),
            hv[..., 0].clip(max=1.0),
        ),
        axis=-1,
    )
    return hsv_to_rgb(hsv)


def is_asleep(index, r2_x, r2_y, k_x, k_y):
    ratio = np.square(r2_x[index]) / max(
        np.square(r2_x[index]) + np.square(r2_y[index]), 1e-8
    )
    freq = ratio * k_x[index] + (1 - ratio) * k_y[index]
    return freq < 0.3  # this is really rough, will be improved later...


def determine_category(index, mov, oob, nonper, shift, slp):
    # aperiodic, asleep, awake, moving, sigh/twitch, out of bed - no apnea detection rn
    # is it periodic? if yes: (is it slp? if yes: asleep, if no: awake) If not...
    # out of bed? if yes: oob, if not...
    # shifting? If yes: mov, if no...
    # mov? if yes: sigh, if no: aper
    if nonper[index]:
        if oob[index]:
            return "oob"
        elif shift[index] > 0.5:
            return "mov"
        elif mov[index]:
            return "sigh"
        else:
            return "aper"
    elif slp:
        return "asleep"
    else:
        return "awake"


def postprocess(dt):
    # 211107_015105 -- this is a good one
    # 211102_003909 -- another goodie. Lots of sleep, some time not in bed at the beginning.
    #               -- No wake up though.
    # 211101_002730 -- Excellent. 5 Sleep cycles visible. One spot not flipped right.
    ######
    # the above ones are all old
    # 220103_232249 - this one has a long gap out of bed, very disturbed sleep, but a few deep sleep and REM blobs
    # 220105_005821 - pretty solid sleep, clear sleep cycles visible relatively evenly spaced, more deep sleep at the start and more REM at the end.
    # dt = "220119_010917"

    gl = sorted(list(Path.cwd().glob(f"sleepypi/run{dt}/*-data.gz")))

    streams = []
    times = []

    # get timezone offset for local START of night, so that DST is handled appropriately
    dt_obj = datetime.strptime(dt, "%y%m%d_%H%M%S")

    nseg = len(gl)
    lengths = np.zeros(nseg + 1, dtype=int)

    for idx in trange(nseg, leave=False):
        with gzip.open(gl[idx], "rb") as f:
            p = pickle.load(f)
            data_stream, tstamps = p
            streams.append(data_stream)
            times.append(tstamps.astype(np.int64) * 50 + 1609459200000)
            lengths[idx + 1] = data_stream.shape[0]

    n = np.concatenate(streams, axis=0)
    npts = n.shape[0]
    # convert times back to epoch time in milliseconds
    all_times = np.concatenate(times, axis=0)

    freqs = np.logspace(0.1, -1.4, 150)  # indices ~50-85 are breathing frequencies

    sm = 81
    pos_x = n[:, 4]
    pos_y = n[:, 5]
    r2_x, r2_y, ab_x, ab_y, k_freq_x, k_freq_y = find_periodic_fit_best(
        n[:, 0], n[:, 1], freqs[40:90]
    )

    mov, oob, nonper = categorize_aper(
        n[:, 0].shape, r2_x, r2_y, ab_x, ab_y, pos_x, pos_y
    )

    p_shift = find_discont_piecewise(pos_x, pos_y, mov)

    exp_k = 30  # 1 second at 30 fps, 6 realtime seconds (most breaths are shorter)
    cond = np.logical_and(nonper, np.logical_not(oob))
    expand = np.concatenate((np.full(exp_k, cond[0]), cond, np.full(exp_k, cond[-1])))
    exp_cs = np.cumsum(expand)
    incl_gaps = exp_cs[2 * exp_k :] - exp_cs[: -2 * exp_k] > 0.5
    # include any gaps less than 2 realtime seceonds (9 frames or less)
    include = medfilt(incl_gaps.astype(np.float32), 19).astype(bool)

    exclude = np.logical_not(include)
    n_excl = np.count_nonzero(exclude)
    cutoff = 4.0 * np.sqrt(np.sum(np.square(n[exclude, :2])) / n_excl)
    omega = 10.0
    fs = 5.0
    freqs = np.logspace(0.1, -1.4, 150)
    widths = omega * fs / (freqs[45:85] * 2 * np.pi)
    mags = cwt_mag_cmpr(n[:, 0], n[:, 1], widths, omega, clipmag=cutoff)

    plt.switch_backend("Agg")

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 0.4), dpi=100)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    # ax.axis("tight")
    ax.axis("off")
    ax.imshow(mags, aspect="auto")
    # plt.show()
    fig.canvas.draw()
    mag_graph = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    mag_graph.resize((40, 320, 3))

    fig, ax = plt.subplots(1, 1, figsize=(3.2, 1.04), dpi=100)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.axis("tight")
    ax.axis("off")
    ax.plot(n[:, 0])
    ax.plot(n[:, 1])
    ax.autoscale(axis="y", tight=True)
    vline = ax.axvline(x=0, c="k", linestyle=":")
    txt_props = dict(boxstyle="round", facecolor="white", alpha=0.7)
    txt = ax.text(
        0.025,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=txt_props,
    )
    tz = pytz.timezone("America/Toronto")

    with open("overlay_text.pkl", "rb") as f:
        overlay = pickle.load(f)

    lg_cs = np.cumsum(lengths)
    gl = sorted(list(Path.cwd().glob(f"sleepypi/run{dt}/*-video.bin")))
    vid = VideoWriter(f"sleepypi/run{dt}/analysis-{dt}.mp4", 20)
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    loop = trange(nseg, leave=False)
    for idx in loop:
        start = lg_cs[idx]
        end = lg_cs[idx + 1]
        indices = np.argwhere(include[start:end])[:, 0]
        loop.set_description(f"{indices.size}/{end-start}")
        with open(gl[idx], "rb") as f:
            shp = [val for val in f.read(3)]
            video = np.frombuffer(f.read(), dtype=np.uint8).reshape(
                [lengths[idx + 1]] + shp
            )
        # maxes = np.max(comps[..., 1], axis=(1, 2))
        # max_exp = maximum_filter1d(maxes, sm,)
        for vdx in indices:
            dt_obj = datetime.fromtimestamp(all_times[vdx + start] * 0.001)
            tzn = tz.tzname(dt_obj)  # returns EST or EDT
            tstr = dt_obj.strftime("%y-%m-%d\n%H:%M:%S ") + tzn
            vline.set_xdata(vdx)
            ax.set_xlim([vdx + start - 60, vdx + start + 60])
            rms = np.sqrt(
                np.mean(
                    np.square(
                        n[max(0, vdx + start - 20) : min(npts, vdx + start + 20), :2]
                    )
                )
            )
            ax.set_ylim([-5.0 * rms, 5.0 * rms])
            txt.set_text(tstr)
            fig.canvas.draw()
            vid_frame = video[vdx, ..., :3].astype(np.float32)
            vf_red_rav = vid_frame[..., 0].ravel()
            if np.count_nonzero(vf_red_rav) > 0:
                vfupper = np.percentile(vf_red_rav, 97.0)
                vid_frame *= 240.0 / vfupper
            velocity = mag_arg_to_rgb_float(video[vdx, ..., -2:])
            slp = is_asleep(vdx + start, r2_x, r2_y, k_freq_x, k_freq_y)
            text = determine_category(
                vdx + start, mov, oob, nonper, np.max(p_shift, axis=1), slp
            )
            graph = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            graph.resize((104, 320, 3))
            text_spot = 1.0 - overlay[text][..., None] * (1.0 - velocity[1:15, 1:73, :])
            velocity[1:15, 1:73, :] = text_spot
            velocity *= 255.9
            img[:96, :160, :] = vid_frame.clip(max=255.0).astype(np.uint8)
            img[:96, 160:, :] = velocity.clip(max=255.0).astype(np.uint8)
            img[96:-40, :, :] = graph
            img[-40:, :, :] = mag_graph
            # which pixels should be lit up in the magnitude graph?
            curr = int((vdx + start) / npts * 320)
            img[-40:, curr, 0] = 255
            vid.add(img)
    vid.close()

    fig.clear()
    plt.close("all")
    print("Video Completed")


if __name__ == "__main__":
    postprocess("230129_230750")

