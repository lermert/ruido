import os
import scipy
import time
import numpy as np
from obspy.signal.invsim import cosine_taper
from scipy.fftpack import fft,ifft,next_fast_len
from obspy.signal.regression import linear_regression
# from numba import jit, float64, int32
# from numba.types import UniTuple

'''
Functions ported / modified from NoisePy (https://noise-python.readthedocs.io/en/latest/)

NoisePy is originally developed by: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
                                    Marine Denolle (mdenolle@fas.harvard.edu)
Several utility functions are modified based on https://github.com/tclements/noise

NoisePy is licensed under MIT License.

MIT License

Copyright (c) 2019 Marine Denolle & Chengxin Jiang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

# @jit(nopython = True)
def moving_ave(A, N):
    '''
    running smooth average for an array.
    PARAMETERS:
    ---------------------
    A: 1-D array of data to be smoothed
    N: integer, it defines the full window length to smooth
    
    RETURNS:
    ---------------------
    B: 1-D array with smoothed data
    '''
   # defines an array with N extra samples at either side
    temp = np.zeros(len(A) + 2 * N)
    # set the central portion of the array to A
    temp[N: -N] = A
    # leading samples: equal to first sample of actual array
    temp[0: N] = temp[N]
    # trailing samples: Equal to last sample of actual array
    temp[-N:] = temp[-N-1]
    # convolve with a boxcar and normalize, and use only central portion of the result
    # with length equal to the original array, discarding the added leading and trailing samples
    B = np.convolve(temp, np.ones(N)/N, mode='same')[N: -N]
    return(B)


def robust_stack(cc_array,epsilon):
    """ 
    this is a robust stacking algorithm described in Palvis and Vernon 2010

    PARAMETERS:
    ----------------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    epsilon: residual threhold to quit the iteration
    RETURNS:
    ----------------------
    newstack: numpy vector contains the stacked cross correlation

    Written by Marine Denolle 
    """
    res  = 9E9  # residuals
    w = np.ones(cc_array.shape[0])
    nstep=0
    newstack = np.median(cc_array,axis=0)
    while res > epsilon:
        stack = newstack
        for i in range(cc_array.shape[0]):
            crap = np.multiply(stack,cc_array[i,:].T)
            crap_dot = np.sum(crap)
            di_norm = np.linalg.norm(cc_array[i,:])
            ri = cc_array[i,:] -  crap_dot*stack
            ri_norm = np.linalg.norm(ri)
            w[i]  = np.abs(crap_dot) /di_norm/ri_norm#/len(cc_array[:,1])
        # print(w)
        w =w /np.sum(w)
        newstack =np.sum( (w*cc_array.T).T,axis=0)#/len(cc_array[:,1])
        res = np.linalg.norm(newstack-stack,ord=1)/np.linalg.norm(newstack)/len(cc_array[:,1])
        nstep +=1
        if nstep>10:
            return newstack, w, nstep
    return newstack, w, nstep

def whiten(data, fft_para):
    '''
    This function takes 1-dimensional timeseries array, transforms to frequency domain using fft, 
    whitens the amplitude of the spectrum in frequency domain between *freqmin* and *freqmax*
    and returns the whitened fft.
    PARAMETERS:
    ----------------------
    data: numpy.ndarray contains the 1D time series to whiten
    fft_para: dict containing all fft_cc parameters such as  
        dt: The sampling space of the `data`
        freqmin: The lower frequency bound
        freqmax: The upper frequency bound
        smooth_N: integer, it defines the half window length to smooth
        freq_norm: whitening method between 'one-bit' and 'RMA'
    RETURNS:
    ----------------------
    FFTRawSign: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds
    '''

    # load parameters
    delta   = fft_para['dt']
    freqmin = fft_para['freqmin']
    freqmax = fft_para['freqmax']
    smooth_N  = fft_para['smooth_N']
    freq_norm = fft_para['freq_norm']

    # Speed up FFT by padding to optimal size for FFTPACK
    if data.ndim == 1:
        axis = 0
    elif data.ndim == 2:
        axis = 1

    Nfft = int(next_fast_len(int(data.shape[axis])))

    Napod = 100
    Nfft = int(Nfft)
    freqVec = scipy.fftpack.fftfreq(Nfft, d=delta)[:Nfft // 2]
    J = np.where((freqVec >= freqmin) & (freqVec <= freqmax))[0]
    low = J[0] - Napod
    if low <= 0:
        low = 1

    left = J[0]
    right = J[-1]
    high = J[-1] + Napod
    if high > Nfft/2:
        high = int(Nfft//2)

    FFTRawSign = scipy.fftpack.fft(data, Nfft, axis=axis)
    # Left tapering:
    if axis == 1:
        FFTRawSign[:, 0:low] *= 0
        FFTRawSign[:, low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:, low:left]))
        # Pass band:
        if freq_norm == 'phase_only':
            FFTRawSign[:, left:right] = np.exp(1j * np.angle(FFTRawSign[:, left:right]))
        elif freq_norm == 'rma':
            for ii in range(data.shape[0]):
                tave = moving_ave(np.abs(FFTRawSign[ii,left:right]), smooth_N)
                FFTRawSign[ii,left:right] = FFTRawSign[ii,left:right] / tave
        # Right tapering:
        FFTRawSign[:,right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:,right:high]))
        FFTRawSign[:,high:Nfft//2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[:,-(Nfft//2)+1:] = np.flip(np.conj(FFTRawSign[:,1:(Nfft//2)]),axis=axis)
    else:
        FFTRawSign[0:low] *= 0
        FFTRawSign[low:left] = np.cos(
            np.linspace(np.pi / 2., np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[low:left]))
        # Pass band:
        if freq_norm == 'phase_only':
            FFTRawSign[left:right] = np.exp(1j * np.angle(FFTRawSign[left:right]))
        elif freq_norm == 'rma':
            tave = moving_ave(np.abs(FFTRawSign[left:right]),smooth_N)
            FFTRawSign[left:right] = FFTRawSign[left:right]/tave
        # Right tapering:
        FFTRawSign[right:high] = np.cos(
            np.linspace(0., np.pi / 2., high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[right:high]))
        FFTRawSign[high:Nfft//2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[-(Nfft//2)+1:] = FFTRawSign[1:(Nfft//2)].conjugate()[::-1]

    return FFTRawSign, Nfft


# @jit(UniTuple(float64, 4)(float64[:], float64[:], float64, int32, float64,
#                              float64, float64, float64, float64),nopython=True)
# def stretching_nb(ref, cur, dv_range, nbtrial, tmin, tmax, fmin, fmax, dt):
    
#     """
#     This function compares the Reference waveform to stretched/compressed current waveforms to get the relative seismic velocity variation (and associated error).
#     It also computes the correlation coefficient between the Reference waveform and the current waveform.
    
#     PARAMETERS:
#     ----------------
#     ref: Reference waveform (np.ndarray, size N)
#     cur: Current waveform (np.ndarray, size N)
#     dv_range: absolute bound for the velocity variation; example: dv=0.03 for [-3,3]% of relative velocity change ('float')
#     nbtrial: number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  ('float')
#     para: vector of the indices of the cur and ref windows on wich you want to do the measurements (np.ndarray, size tmin*delta:tmax*delta)
#     For error computation, we need parameters:
#         fmin: minimum frequency of the data
#         fmax: maximum frequency of the data
#         tmin: minimum time window where the dv/v is computed 
#         tmax: maximum time window where the dv/v is computed 
#     RETURNS:
#     ----------------
#     dv: Relative velocity change dv/v (in %)
#     cc: correlation coefficient between the reference waveform and the best stretched/compressed current waveform
#     cdp: correlation coefficient between the reference waveform and the initial current waveform
#     error: Errors in the dv/v measurements based on Weaver et al (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3)

#     Note: The code first finds the best correlation coefficient between the Reference waveform and the stretched/compressed current waveform among the "nbtrial" values. 
#     A refined analysis is then performed around this value to obtain a more precise dv/v measurement .

#     Originally by L. Viens 04/26/2018 (Viens et al., 2018 JGR)
#     modified by Chengxin Jiang & Laura Ermert (numba)
#     """ 
#     # load common variables from dictionary
#     # twin = para['twin']
#     # freq = para['freq']
#     # dt   = para['dt']
#     # tmin = np.min(twin)
#     # tmax = np.max(twin)
#     # fmin = np.min(freq)
#     # fmax = np.max(freq)
#     tvec = np.arange(tmin,tmax,dt)
#     # make useful one for measurements
#     dvmin = -np.abs(dv_range)
#     dvmax = np.abs(dv_range)
#     Eps = 1+(np.linspace(dvmin, dvmax, nbtrial))
#     cof = np.zeros(Eps.shape,dtype=np.float32)

#     # Set of stretched/compressed current waveforms
#     for ii in range(len(Eps)):
#         nt = tvec*Eps[ii]
#         s = np.interp(x=tvec, xp=nt, fp=cur)
#         waveform_ref = ref
#         waveform_cur = s
#         cof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]
#         if np.isnan(cof[ii]):
#             cof[ii] = -100.

#     cdp = np.corrcoef(cur, ref)[0, 1] # correlation coefficient between the reference and initial current waveforms

#     # find the maximum correlation coefficient
#     imax = np.argmax(cof)
#     if imax >= len(Eps)-2:
#         imax = imax - 2
#     if imax <= 2:
#         imax = imax + 2

#     # Proceed to the second step to get a more precise dv/v measurement
#     dtfiner = np.linspace(Eps[imax-2], Eps[imax+2], 100)
#     ncof    = np.zeros(dtfiner.shape,dtype=np.float32)
#     for ii in range(len(dtfiner)):
#         nt = tvec*dtfiner[ii]
#         s = np.interp(x=tvec, xp=nt, fp=cur)
#         waveform_ref = ref
#         waveform_cur = s
#         ncof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]
#         if np.isnan(ncof[ii]):
#             ncof[ii] = -100.

#     cc = np.max(ncof) # Find maximum correlation coefficient of the refined  analysis
#     dv = 100. * dtfiner[np.argmax(ncof)]-100 # Multiply by 100 to convert to percentage (Epsilon = -dt/t = dv/v)

#     # Error computation based on Weaver et al (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3)
#     T = 1 / (fmax - fmin)
#     X = cc
#     wc = np.pi * (fmin + fmax)
#     t1 = tmin
#     t2 = tmax
#     error = 100*(np.sqrt(1-X**2)/(2*X)*np.sqrt((6* np.sqrt(np.pi/2)*T)/(wc**2*(t2**3-t1**3))))
#     cc = float64(cc)

#     return (dv, error, cc, cdp)


def stretching_vect(ref, cur, dv_range, nbtrial, para):
    
    """
    This function compares the Reference waveform to stretched/compressed current waveforms to get the relative seismic velocity variation (and associated error).
    It also computes the correlation coefficient between the Reference waveform and the current waveform.
    
    PARAMETERS:
    ----------------
    ref: Reference waveform (np.ndarray, size N)
    cur: Current waveform (np.ndarray, size N)
    dv_range: absolute bound for the velocity variation; example: dv=0.03 for [-3,3]% of relative velocity change ('float')
    nbtrial: number of stretching coefficient between dvmin and dvmax, no need to be higher than 100  ('float')
    para: vector of the indices of the cur and ref windows on wich you want to do the measurements (np.ndarray, size tmin*delta:tmax*delta)
    For error computation, we need parameters:
        fmin: minimum frequency of the data
        fmax: maximum frequency of the data
        tmin: minimum time window where the dv/v is computed 
        tmax: maximum time window where the dv/v is computed 
    RETURNS:
    ----------------
    dv: Relative velocity change dv/v (in %)
    cc: correlation coefficient between the reference waveform and the best stretched/compressed current waveform
    cdp: correlation coefficient between the reference waveform and the initial current waveform
    error: Errors in the dv/v measurements based on Weaver et al (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3)

    Note: The code first finds the best correlation coefficient between the Reference waveform and the stretched/compressed current waveform among the "nbtrial" values. 
    A refined analysis is then performed around this value to obtain a more precise dv/v measurement .

    Originally by L. Viens 04/26/2018 (Viens et al., 2018 JGR)
    modified by Chengxin Jiang
    modified by Laura Ermert: vectorized version
    """ 
    # load common variables from dictionary
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    n_dec = len(str(1./dt))
    print(np.min(twin))
    tmin = round(np.min(twin), n_dec)
    tmax = round(np.max(twin), n_dec)
    print(tmin, tmax)
    fmin = np.min(freq)
    fmax = np.max(freq)
    tvec = np.arange(tmin, tmax + dt/2., dt)

    # make useful one for measurements
    dvmin = -np.abs(dv_range)
    dvmax = np.abs(dv_range)
    Eps = 1 + (np.linspace(dvmin, dvmax, nbtrial))
    #cof = np.zeros(Eps.shape,dtype=np.float32)
    cdp = np.corrcoef(cur, ref)[0, 1] # correlation coefficient between the reference and initial current waveforms
    waveforms = np.zeros((nbtrial + 1, len(ref)))
    waveforms[0, :] = ref
    # nt = np.array(nbtrial * [tvec])
    # nt = np.multiply(nt.transpose(), Eps).transpose()
    # print(waveforms.shape, tvec.shape, cur.shape, ref.shape)
    # Set of stretched/compressed current waveforms
    for ii in range(nbtrial):
        nt = tvec * Eps[ii]
        s = np.interp(x=tvec, xp=nt, fp=cur)
        #waveform_cur = s
        #cof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]
        waveforms[ii + 1, :] = s
    cof = np.corrcoef(waveforms)[0][1:]
    
    # find the maximum correlation coefficient
    imax = np.nanargmax(cof)
    if imax >= len(Eps)-2:
        imax = imax - 2
    if imax < 2:
        imax = imax + 2

    # Proceed to the second step to get a more precise dv/v measurement
    dtfiner = np.linspace(Eps[imax-2], Eps[imax+2], nbtrial)
    #ncof    = np.zeros(dtfiner.shape,dtype=np.float32)
    waveforms = np.zeros((nbtrial + 1, len(ref)))
    waveforms[0, :] = ref
    for ii in range(len(dtfiner)):
        nt = tvec * dtfiner[ii]
        s = np.interp(x=tvec, xp=nt, fp=cur)
        #waveform_ref = ref
        #waveform_cur = s
        #ncof[ii] = np.corrcoef(waveform_ref, waveform_cur)[0, 1]
        waveforms[ii + 1, :] = s
    #print(waveforms)
    ncof = np.corrcoef(waveforms)[0][1: ]
    #print(ncof)
    cc = np.max(ncof) # Find maximum correlation coefficient of the refined  analysis
    dv = 100. * dtfiner[np.argmax(ncof)] - 100 # Multiply by 100 to convert to percentage (Epsilon = -dt/t = dv/v)

    # Error computation based on Weaver et al (2011), On the precision of noise-correlation interferometry, Geophys. J. Int., 185(3)
    T = 1 / (fmax - fmin)
    X = cc
    wc = np.pi * (fmin + fmax)
    t1 = np.min([tmin, tmax])
    t2 = np.max([tmin, tmax])
    #print(cc, X)
    error = 100*(np.sqrt(1-X**2)/(2*X)*np.sqrt((6* np.sqrt(np.pi/2)*T)/(wc**2*(t2**3-t1**3))))

    return dv, error, cc, cdp


def dtw_dvv(ref, cur, para, maxLag, b, direction, return_warp_path=True):
    """
    Dynamic time warping for dv/v estimation.
    
    PARAMETERS:
    ----------------
    ref : reference signal (np.array, size N)
    cur : current signal (np.array, size N)
    para: dict containing useful parameters about the data window and targeted frequency
    maxLag : max number of points to search forward and backward. 
            Suggest setting it larger if window is set larger.
    b : b-value to limit strain, which is to limit the maximum velocity perturbation. 
            See equation 11 in (Mikesell et al. 2015)
    direction: direction to accumulate errors (1=forward, -1=backward)
    RETURNS:
    ------------------
    -m0 : estimated dv/v
    em0 : error of dv/v estimation
    
    Original by Di Yang
    Last modified by Dylan Mikesell (25 Feb. 2015)
    Translated to python by Tim Clements (17 Aug. 2018)    
    """
    twin = para['twin']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    tvect = np.arange(tmin,tmax,dt)

    # setup other parameters
    npts = len(ref) # number of time samples
    
    # compute error function over lags, which is independent of strain limit 'b'.
    err = computeErrorFunction( cur, ref, npts, maxLag ) 
    
    # direction to accumulate errors (1=forward, -1=backward)
    dist  = accumulateErrorFunction( direction, err, npts, maxLag, b )
    stbar = backtrackDistanceFunction( -1*direction, dist, err, -maxLag, b )
    stbarTime = stbar * dt   # convert from samples to time
    
    # evaluate cc before and after warping
    ccoeff_original = np.corrcoef(ref, cur)[0][1]
    ixs_applyshift = np.arange(len(cur)) - stbar
    ixs_applyshift = np.clip(ixs_applyshift, 0, len(cur)-1)
    ixs_applyshift = np.array(ixs_applyshift, dtype=np.int)
    warped = cur[ixs_applyshift]
    ccoeff_shifted = np.corrcoef(ref, warped)[0][1]
    print(ccoeff_original, ccoeff_shifted)

    if return_warp_path:
        return(stbarTime, dist, ccoeff_original, ccoeff_shifted)
    else:
        # cut the first and last 5% for better regression
        indx = np.where((tvect>=0.05*npts*dt) & (tvect<=0.95*npts*dt))[0]

        # linear regression to get dv/v
        if npts >2:

            # weights
            w = np.ones(npts)
            #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
            m0, em0 = linear_regression(tvect.flatten()[indx], stbarTime.flatten()[indx], w.flatten()[indx], intercept_origin=True)

        else:
            print('not enough points to estimate dv/v for dtw')
            m0=0;em0=0
        
        return m0*100,em0*100,dist


def mwcs_dvv(ref, cur, moving_window_length, slide_step, para, smoothing_half_win=5):
    """
    Moving Window Cross Spectrum method to measure dv/v (relying on phi=2*pi*f*t in freq domain)

    PARAMETERS:
    ----------------
    ref: Reference waveform (np.ndarray, size N)
    cur: Current waveform (np.ndarray, size N)
    moving_window_length: moving window length to calculate cross-spectrum (np.float, in sec)
    slide_step: steps in time to shift the moving window (np.float, in seconds)
    para: a dict containing parameters about input data window and frequency info, including 
        delta->The sampling rate of the input timeseries (in Hz)
        window-> The target window for measuring dt/t
        freq-> The frequency bound to compute the dephasing (in Hz)
        tmin: The leftmost time lag (used to compute the "time lags array")
    smoothing_half_win: If different from 0, defines the half length of the smoothing hanning window.
    
    RETURNS:
    ------------------
    time_axis: the central times of the windows. 
    delta_t: dt
    delta_err:error 
    delta_mcoh: mean coherence
    
    Copied from MSNoise (https://github.com/ROBelgium/MSNoise/tree/master/msnoise)
    Modified by Chengxin Jiang
    """
    # common variables
    twin = para['twin']
    freq = para['freq']
    dt   = para['dt']
    tmin = np.min(twin)
    tmax = np.max(twin)
    fmin = np.min(freq)
    fmax = np.max(freq)
    tvect = np.arange(tmin,tmax,dt)

    # parameter initialize
    delta_t = []
    delta_err = []
    delta_mcoh = []
    time_axis = []

    # info on the moving window
    window_length_samples = np.int(moving_window_length/dt)
    padd = int(2 ** (nextpow2(window_length_samples) + 2))
    count = 0
    tp = cosine_taper(window_length_samples, 0.15)

    minind = 0
    maxind = window_length_samples

    # loop through all sub-windows
    while maxind <= len(ref):
        cci = cur[minind:maxind]
        cci = scipy.signal.detrend(cci, type='linear')
        cci *= tp

        cri = ref[minind:maxind]
        cri = scipy.signal.detrend(cri, type='linear')
        cri *= tp

        minind += int(slide_step/dt)
        maxind += int(slide_step/dt)

        # do fft
        fcur = scipy.fftpack.fft(cci, n=padd)[:padd // 2]
        fref = scipy.fftpack.fft(cri, n=padd)[:padd // 2]

        fcur2 = np.real(fcur) ** 2 + np.imag(fcur) ** 2
        fref2 = np.real(fref) ** 2 + np.imag(fref) ** 2

        # get cross-spectrum & do filtering
        X = fref * (fcur.conj())
        if smoothing_half_win != 0:
            dcur = np.sqrt(smooth(fcur2, window='hanning',half_win=smoothing_half_win))
            dref = np.sqrt(smooth(fref2, window='hanning',half_win=smoothing_half_win))
            X = smooth(X, window='hanning',half_win=smoothing_half_win)
        else:
            dcur = np.sqrt(fcur2)
            dref = np.sqrt(fref2)

        dcs = np.abs(X)

        # Find the values the frequency range of interest
        freq_vec = scipy.fftpack.fftfreq(len(X) * 2, dt)[:padd // 2]
        index_range = np.argwhere(np.logical_and(freq_vec >= fmin,freq_vec <= fmax))

        # Get Coherence and its mean value
        coh = getCoherence(dcs, dref, dcur)
        mcoh = np.mean(coh[index_range])

        # Get Weights
        w = 1.0 / (1.0 / (coh[index_range] ** 2) - 1.0)
        w[coh[index_range] >= 0.99] = 1.0 / (1.0 / 0.9801 - 1.0)
        w = np.sqrt(w * np.sqrt(dcs[index_range]))
        w = np.real(w)

        # Frequency array:
        v = np.real(freq_vec[index_range]) * 2 * np.pi

        # Phase:
        phi = np.angle(X)
        phi[0] = 0.
        phi = np.unwrap(phi)
        phi = phi[index_range]

        # Calculate the slope with a weighted least square linear regression
        # forced through the origin; weights for the WLS must be the variance !
        m, em = linear_regression(v.flatten(), phi.flatten(), w.flatten())
        delta_t.append(m)

        # print phi.shape, v.shape, w.shape
        e = np.sum((phi - m * v) ** 2) / (np.size(v) - 1)
        s2x2 = np.sum(v ** 2 * w ** 2)
        sx2 = np.sum(w * v ** 2)
        e = np.sqrt(e * s2x2 / sx2 ** 2)

        delta_err.append(e)
        delta_mcoh.append(np.real(mcoh))
        time_axis.append(tmin+moving_window_length/2.+count*slide_step)
        count += 1

        del fcur, fref
        del X
        del freq_vec
        del index_range
        del w, v, e, s2x2, sx2, m, em

    if maxind > len(cur) + int(slide_step/dt):
        print("The last window was too small, but was computed")

    # ensure all matrix are np array
    delta_t = np.array(delta_t)
    delta_err = np.array(delta_err)
    delta_mcoh = np.array(delta_mcoh)
    time_axis  = np.array(time_axis)

    # ready for linear regression
    delta_mincho = 0.65
    delta_maxerr = 0.1
    delta_maxdt  = 0.1
    indx1 = np.where(delta_mcoh>delta_mincho)
    indx2 = np.where(delta_err<delta_maxerr)
    indx3 = np.where(delta_t<delta_maxdt)

    #-----find good dt measurements-----
    indx = np.intersect1d(indx1,indx2)
    indx = np.intersect1d(indx,indx3)

    if len(indx) >2:

        #----estimate weight for regression----
        w = 1/delta_err[indx]
        w[~np.isfinite(w)] = 1.0

        #---------do linear regression-----------
        #m, a, em, ea = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=False)
        m0, em0 = linear_regression(time_axis[indx], delta_t[indx], w, intercept_origin=True)
    
    else:
        print('not enough points to estimate dv/v for mwcs')
        m0=0;em0=0

    return -m0*100,em0*100
