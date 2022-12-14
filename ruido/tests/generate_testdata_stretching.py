# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.signal import hann
from obspy import UTCDateTime

x = np.linspace(0., 100., 10000)


a = np.sin(x*4)
win = np.zeros(len(x))
win[3000:4000] = hann(1000)
a *= win

t_stretch = np.linspace(0, 99.8, 10000)  # 0.5 % perturbation
b = np.interp(x, t_stretch, a)

t_stretch = np.linspace(0, 99.0, 10000)
c = np.interp(x, t_stretch, a)



plt.plot(a); plt.plot(b); plt.plot(c)
plt.show()

t0 = UTCDateTime().timestamp
t1 = t0 + 86400.0
t2 = t1 + 86400.0
timestamps = np.array([t0 +  i * 86400.0 for i in range(20)])
data = np.array([a, a, a, a, a, a, a, a, a, b, c, a, a, a, a, a, a, a, a, a])

fs = 100.


f = h5py.File("testdata_stretching.h5", "w")

g = f.create_group("corr_windows")
g.create_dataset("data", data=data)
g.create_dataset("timestamps", data=timestamps)
st = f.create_dataset("stats", data=())

st.attrs["sampling_rate"] = fs
f.flush()
f.close()
