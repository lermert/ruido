# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filtering functions ported, modified and extended
# from obspy (https://github.com/obspy/obspy)
# obspy is licensed under GNU LGPL V3, as is ruido.
# Modifications:
# - do not apply filters, but only return coefficients
# - Chebysheff bandpass filter
#
#
# Filename: filter.py
#  Purpose: Various Seismogram Filtering Functions
#   Author: Tobias Megies, Moritz Beyreuther, Yannik Behr
#    Email: tobias.megies@geophysik.uni-muenchen.de
#
# Copyright (C) 2009 Tobias Megies, Moritz Beyreuther, Yannik Behr
# --------------------------------------------------------------------

import warnings
import pycwt
from scipy.signal import iirfilter
try:
    from scipy.signal import zpk2sos
except ImportError:
    from obspy.signal._sosfilt import _zpk2sos as zpk2sos
from scipy.signal import cheb2ord, cheby2, hann, tukey
import numpy as np

"""
Various Seismogram Filtering Functions from obspy

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
def get_window(t_mid, hw, lag, window_type, alpha=0.2):

    # find closest sample to center of window
    ix_c = np.argmin(abs(lag - t_mid))
    delta = np.diff(lag).mean()
    hw_samples = int(round(hw / delta))

    ix_0 = ix_c - hw_samples
    ix_1 = ix_c + hw_samples + 1  # account for one central sample

    win = np.zeros(lag.shape)
    if window_type == "hann":
        win[ix_0: ix_1] = hann(ix_1 - ix_0)
    elif window_type == "tukey":
        win[ix_0: ix_1] = tukey(ix_1 - ix_0, alpha=alpha)
    elif window_type == "boxcar":
        win[ix_0: ix_1] = 1.0
    return(win)

def moving_average(self, a, n=3, rank=0):
    if rank != 0:
        raise ValueError("Call this function only on one process")
    ret = np.cumsum(a, dtype=np.complex)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

def cheby2_lowpass(df, freq, maxorder=8):
    # From obspy
    nyquist = df * 0.5
    # rp - maximum ripple of passband, rs - attenuation of stopband
    rp, rs, order = 1, 96, 1e99
    ws = freq / nyquist  # stop band frequency
    wp = ws  # pass band frequency
    # raise for some bad scenarios
    if ws > 1:
        ws = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    while True:
        if order <= maxorder:
            break
        wp = wp * 0.99
        order, wn = cheb2ord(wp, ws, rp, rs, analog=0)
    print("Pass band, stop band: ", wp * nyquist, ws * nyquist)
    (z, p, k) = cheby2(order, rs, wn, btype='low', analog=0, output='zpk')
    return zpk2sos(z, p, k)


def cheby2_bandpass(df, freq0, freq1, maxorder=8):
    nyquist = df * 0.5
    # rp - maximum ripple of passband, rs - attenuation of stopband
    rp, rs, order = 1, 60, 1e99
    ws = [freq0 / nyquist, freq1 / nyquist]  # stop band frequency
    wp =  ws.copy()  # pass band frequency
    # raise for some bad scenarios
    if ws[1] > 1:
        ws[1] = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    while True:
        if order <= maxorder:
            break
        wp[1] = wp[1] * 0.99
        wp[0] = wp[0] * 1.01
        order, wn = cheb2ord(wp, ws, rp, rs, analog=0)
    print("Pass band, stop band: ", wp[0] * nyquist, wp[1] * nyquist, 
          ws[0] * nyquist, ws[1] * nyquist)
    (z, p, k) = cheby2(order, rs, wn, btype='bandpass', analog=0, output='zpk')
    return zpk2sos(z, p, k)


def cwt_bandpass(data, freqmin, freqmax, df, dj=1 / 12,
                  mother_wavelet='morlet'):
    """
    Bandpass filter by continuous wavelet transform and frequency selection
    Based on the pycwt module
    :type data: numpy ndarray
    :param data: input trace
    :type freqmin: float
    :param freqmin: lower frequency
    :type freqmax: float
    :param freqmax: upper frequency
    :type df: float
    :param df: Sampling rate in Hz
    :param dj: Interval between scales (see :func: `pycwt.cwt`)
    :param s0: Lowest scale (see :func: `pycwt.cwt`)
    :param mother_wavelet: Mother wavelet
    
    """
    dt = 1. / df
    if mother_wavelet == "morlet":
        mother_wavelet = pycwt.wavelet.Morlet(f0=6.0)
    cwt, scales, freqs, cone_of_influence, _, _ = pycwt.cwt(data, dt, dj, wavelet=mother_wavelet, s0=dt)

    ix_freq = np.where((freqs >= freqmin) & (freqs <= freqmax))[0]
    return np.real(pycwt.icwt(cwt[ix_freq], scales[ix_freq], dt, dj, mother_wavelet))


def bandpass(freqmin, freqmax, df, corners=4):
    """
    Butterworth-Bandpass Filter.

    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high > 1:
        high = 1.0
        msg = "Selected high corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    return sos


def lowpass(freq, df, corners=4):
    """
    Butterworth-Lowpass Filter.

    Filter data removing data over certain frequency ``freq`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        f = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    z, p, k = iirfilter(corners, f, btype='lowpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    return sos


def highpass(freq, df, corners=4):
    """
    Butterworth-Highpass Filter.

    Filter data removing data below certain frequency ``freq`` using
    ``corners`` corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        msg = "Selected corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    return sos
