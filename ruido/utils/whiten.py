import numpy as np
from scipy.fftpack import next_fast_len


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
    N = int(N)
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


def whiten(timeseries, sampling_rate, freqmin, freqmax,
           n_smooth=1, n_taper=50, n_taper_min=5):
    
    dt = 1. / sampling_rate
    nfft = next_fast_len(2 * len(timeseries))
    spec = np.fft.rfft(timeseries, nfft)
    freq = np.fft.rfftfreq(nfft, d=dt)
    ix0 = np.argmin(np.abs(freq - freqmin))
    ix1 = np.argmin(np.abs(freq - freqmax))
    
    if ix1 + n_taper >= nfft:
        if nfft - ix1 >= n_taper_min:
            ix11 = nfft
        else:
            raise ValueError("Not enough space to taper, choose a lower freq_max")
    else:
        ix11 = ix1 + n_taper


    if ix0 - n_taper < 0:
        if ix0 >= n_taper_min:
            ix00 = 0
        else:
            raise ValueError("Not enough space to taper, choose a higher freq_min")
    else:
        ix00 = ix0 - n_taper


    spec_out = spec.copy()
    spec_out[0: ix00] = 0.0 + 0.0j
    spec_out[ix11:] = 0.0 + 0.0j

    if n_smooth <= 1:
        spec_out[ix00: ix11] = np.exp(1.j * np.angle(spec_out[ix00: ix11]))
    else:
        spec_out[ix00: ix11] /= moving_ave(np.abs(spec_out[ix00: ix11]), n_smooth)

    x = np.linspace(np.pi / 2., np.pi, ix0 - ix00)
    spec_out[ix00: ix0] *= (np.cos(x) ** 2)


    x = np.linspace(0., np.pi / 2., ix11 - ix1)
    # print(np.cos(x).shape, ix1, ix11)
    spec_out[ix1: ix11] *= (np.cos(x) ** 2)

    return(spec_out, nfft)

