import numpy as np
# routines for cross-correlation time shift and polarization analysis

def cc_timeshift(reference, trace, fs):

    # lag vector
    npts = len(trace)

    if npts % 2 == 1:  # odd
        lag = np.linspace(-(npts - 1) // 2, (npts - 1) // 2, npts)
    else:  # even
        lag = np.linspace(-npts // 2, (npts - 1) // 2, npts)

    ccorr = np.correlate(trace, reference, "same")
    cc_t = lag[np.argmax(ccorr)]
    ccoeff_best = np.max(ccorr)
    ccoeff_start = np.corrcoeff(trace, reference)[0][0]

    return cc_t, ccoeff_best, ccoeff_start


