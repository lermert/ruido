import numpy as np
import os
from obspy import UTCDateTime


indir = ""
files = []
durations = []
adapt_labels = []


for file in files:

    clusters = np.load(os.path.join(indir, file))
    ctemp = clusters[1].copy()
    
    for ixal, al in enumerate(adapt_labels):
        t0 = UTCDateTime(durations[ixal][0]).timestamp
        t1 = UTCDateTime(durations[ixal][1]).timestamp
        ixt = np.intersect1d(np.where(clusters[0] < t1), np.where(clusters[0] >= t0))
        ix = np.where(ctemp[ixt] == al[0])
        ix2 = np.where(ctemp[ixt] == al[1])
        ctemp[ix] = al[1]
        ctemp[ix2] = al[0]

    clusters[1, :] = ctemp
    np.save(os.path.join(indir, file), clusters)
