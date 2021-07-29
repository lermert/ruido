import numpy as np
# routines for cross-correlation time shift and polarization analysis

def cc_timeshift(reference, trace, fs):

    # lag vector
    npts = len(trace)

    if npts % 2 == 1:  # odd
        lag = np.linspace(-(npts - 1) // 2 / fs, (npts - 1) // 2 / fs, npts)
    else:  # even
        lag = np.linspace(-npts // 2 / fs, (npts - 1) // 2 / fs, npts)
    print(lag)
    ccorr = np.correlate(trace, reference, "same")
    shift = np.argmax(ccorr)
    cc_t = lag[shift]
    ccoeff_best = np.max(ccorr) / (np.sqrt(np.sum(trace ** 2)) * np.sqrt(np.sum(reference ** 2)))
    ccoeff_start = np.corrcoef(trace, reference)[1][0]

    return cc_t, ccoeff_best, ccoeff_start