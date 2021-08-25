import numpy as np
from obspy import read
from scipy.signal import hilbert, hann
from obspy.geodetics import gps2dist_azimuth
import matplotlib.pyplot as plt
from obspy import Trace, Stream
from math import degrees, atan2

# function to obtain polarization parameter of 1 window
def polar(e, n, z):
    # simple polarization analysis on the basis of time series
    # covariance
    C = np.cov(np.array([z, n, e]))
    # eigenvalue decomposition
    evl, evc = np.linalg.eig(C)
    evc = np.transpose(evc)
   # print(C)

    # need to sort eigenvalues & eigenvectors first
    l0 = max(evl)
    v0 = evc[np.argmax(evl)]
    #print(v0)
    ixmid = [i for i in [0, 1, 2] if i not in [np.argmax(evl), np.argmin(evl)]][0]
    l1 = evl[ixmid]
    v1 = evc[ixmid]
    l2 = min(evl)
    v2 = evc[np.argmin(evl)]


    # get planarity
    p = 1. - (2. * l2) / (l0 + l1)
    # linearity
    l = (l0 - l2) / np.sum(evl)

    # azimuth
    az = degrees(atan2(v0[2], v0[1]))
    
    eve = np.sqrt(v0[2] ** 2 + v0[1] ** 2)
    incidence = degrees(atan2(eve, v0[0]))
    if az < 0.0:
        az = 360.0 + az
    if incidence < 0.0:
        incidence += 180.0
    if incidence > 90.0:
        incidence = 180.0 - incidence
        if az > 180.0:
            az -= 180.0
        else:
            az += 180.0
    if az > 180.0:
        az -= 180.0

    return(p, l, az, incidence)

def polar_vidale(e, n, z):
    # polarization analysis according to John Vidale's 1986 paper
    # this is in the complex domain and includes searching the optimal angle for the 
    # first eigenvector in the complex domain
    # to resolve its 180 degree ambiguity

    # analytic signal
    E = hilbert(e)
    N = hilbert(n)
    Z = hilbert(z)

    # covariance matrix
    C = np.cov(np.array([N, E, Z]))

    # eigenvalues and vectors
    l, v = np.linalg.eig(C)
    v = np.transpose(v)
    l = l.real
    # need to sort these
    l0 = max(l)
    v0 = v[np.argmax(l)]
    ixmid = [i for i in [0, 1, 2] if i not in [np.argmax(l), np.argmin(l)]][0]
    l1 = l[ixmid]
    v1 = v[ixmid]
    l2 = min(l)
    v2 = v[np.argmin(l)]

    # get optimal angle of first eigenvector
    a = np.linspace(0., np.pi, 360)
    x = np.zeros(360)
    for i, alph in enumerate(a):
        cis = np.cos(alph) + 1.j * np.sin(alph)
        x[i] = np.sqrt((np.real(v0[0] * cis) ** 2 + np.real(v0[1] * cis) ** 2 + np.real(v0[2] * cis) **2))
    ixm = np.argmax(x)
    X = x[ixm]
    # rotate by the angle we found
    v0_rot = v0 * (np.cos(a[ixm]) + 1.j * np.sin(a[ixm]))
    # ellipticity
    p = np.sqrt(1. - X ** 2) / X
    # strength of polarization after Vidale
    ps = 1. - (l1 + l2) / l0
    # azimuth
    #print(v0_rot[1].real, v0_rot[0].real)
    az = np.arctan2(v0_rot[1].real, v0_rot[0].real)
    az = np.rad2deg(az)
    eve = np.sqrt(v0[2].real ** 2 + v0[1].real ** 2)
    incidence = degrees(atan2(eve, v0[0].real))
    if az < 0.0:
        az = 360.0 + az
    if incidence < 0.0:
        incidence += 180.0
    if incidence > 90.0:
        incidence = 180.0 - incidence
        if az > 180.0:
            az -= 180.0
        else:
            az += 180.0
    if az > 180.0:
        az -= 180.0
    
    return(p, ps, az, incidence)


def ma(input_array, points):  # moving average
    output_array = input_array.copy()
    for i in range(len(input_array)):
        if i > points // 2 and len(input_array) > (i - points // 2 + points):
            output_array[i] = np.sum(input_array[i - points // 2: i - points // 2 + points]) / points
        elif i <= points // 2:
            output_array[i] = np.sum(input_array[0: i - points // 2 + points]) / (points - (points // 2 - i))
        elif len(input_array) <= (i - points // 2 + points):
            output_array[i] = np.sum(input_array[i - points // 2:]) / (len(input_array) - (i - points // 2))
    return(output_array)

def ma2(input_array, points):
    output_array = np.zeros(input_array.shape)
    temp = np.convolve(input_array, np.ones(points), 'valid') / points
    difflen = len(output_array) - len(temp)
    output_array[difflen//2: difflen//2 + len(temp)] = temp
    return(output_array)

# function to walk over slidling windows of a time series get polarization parameters, and average them
def pol_win(tr, winlen, step=None, offset=0, method="timedomain", moving_average=3):


    if method == "timedomain":
        polmethod = polar
    else:
        polmethod = polar_vidale

    if step is None:
        step = winlen

    azs = []
    ts = []
    ps = []
    ls = []
    amps = []
    pgvs = []
    for win in tr.slide(offset=offset, window_length=winlen, step=step):
        #print(win[0].stats.starttime)
       # win.plot()
        winc = win.copy()
        #winc.taper(0.05)
        p, l, az, inc = polmethod(winc[0].data, winc[1].data, winc[2].data)
        azs.append(az)
        ts.append(win[0].stats.starttime - tr[0].stats.starttime)
        ps.append(p)
        ls.append(l)
        amps.append(np.sum(np.sqrt(winc[0].data ** 2 + winc[1].data ** 2 + winc[2].data ** 2)))
        pgvs.append(np.max(np.concatenate((winc[0].data, winc[1].data, winc[2].data))))
    if moving_average > 0:
        ls = ma(ls, points=moving_average)
        ps = ma(ps, points=moving_average)
        azs = ma(azs, points=moving_average)
        amps = ma(amps, points=moving_average)
        pgvs = ma(pgvs, points=moving_average)
    return(ts, azs, ps, ls, amps, pgvs)


def basic_example(wave, noiseampl=0.05):
    # or try it out on synthetic example
    f = np.linspace(0., 100. * np.pi, 5000)
    taper = np.zeros(5000)
    taper[1200:2400] = hann(1200)
    # "Rayleigh wave"
    if wave == "rayleigh":
        x = np.cos(f)
        x *= taper
        x += noiseampl * (np.random.random(5000) - 0.5)
        z = np.cos(f - np.pi / 2.)
        z *= taper
        z += noiseampl * (np.random.random(5000) - 0.5)
        y = np.zeros(5000)
        y += noiseampl * (np.random.random(5000) - 0.5)

    elif wave == "love":
        y = np.cos(f)
        y *= taper
        y += noiseampl * (np.random.random(5000) - 0.5)
        z = np.zeros(5000)
        z += noiseampl * (np.random.random(5000) - 0.5)
        x = np.zeros(5000)
        x += noiseampl * (np.random.random(5000) - 0.5)
    elif wave == "love2":
        y = np.cos(f) * -2
        y *= taper
        y += noiseampl * (np.random.random(5000) - 0.5)
        z = np.zeros(5000)
        z += noiseampl * (np.random.random(5000) - 0.5)
        x = np.cos(f) * -1
        x *= taper
        x += noiseampl * (np.random.random(5000) - 0.5)
    e = Trace(data=x)
    e.stats.channel="LHE"
    e.stats.sampling_rate = 5.
    n = Trace(data = y)
    n.stats.channel="LHN"
    n.stats.sampling_rate = 5.
    z = Trace(data=z)
    z.stats.channel="LHZ"
    z.stats.sampling_rate = 5.

    tr = Stream()
    tr += e
    tr += n
    tr += z

    return tr


def data_example():
    # # try it out on test event in Mexico, recorded in France
    # # actual azimuth, back azimuth:
    dist, az, baz = gps2dist_azimuth(45.279, 4.542, 15.886, -96.008)
    print("Known azimuth epicenter - station: ", az)
    tr = read("testevent_polar/G.SSB.00.LH*.2020-06-23T12:09:04.*.sac")
    print(tr)
    tr.trim(starttime = tr[0].stats.starttime + 12000)
    tr.trim(endtime = tr[0].stats.starttime + 11000)
    tr.taper(0.05)
    tr.filter("bandpass", freqmin=0.05, freqmax=0.1, corners=4, zerophase=True)
    #tr.plot()
    tr.sort(keys=["channel"])
    return tr


if __name__ == "__main__":
    window_length = 20.
    window_step = 20
    moving_average=3
    example = "synthetic-rayleigh"
    tr = basic_example("rayleigh")#data_example()
    #b=polarization.polarization_analysis(tr,20.0,1.0,10,12,tr.traces[0].meta.starttime,tr.traces[0].meta.endtime,False,"flinn")
    #print(b)
    tr_keep = tr.copy()
    xwin = [200, 600] #[0, 4600] #[200, 600]#[

    # p, l, az = polar_vidale(tr[0].data, tr[1].data, tr[2].data)
    # print("Azimuth w/ frequency domain algorithm: ", az)

    # tr = tr_keep.copy()
    # p, l, az = polar(tr[0].data, tr[1].data, tr[2].data)
    # print("Azimuth w/ time domain algorithm: ", az)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(211)
    t = np.linspace(0, tr[0].stats.npts * tr[0].stats.delta, tr[0].stats.npts)
    ax.plot(t, tr[0].data, "k", alpha=0.5)
    ax.plot(t, tr[1].data, "k", alpha=0.5)
    ax.plot(t, tr[2].data, "k", alpha=0.5)
    ax.tick_params(axis='y', labelcolor="k")
    ax.set_ylabel("Seismic timeseries", color="k")
    tr = tr_keep.copy()
    ts, azs, ps, ls, ams, pgvs = pol_win(tr, window_length, step=window_step, moving_average=moving_average)
    ax2 = ax.twinx()
    azs = np.array(azs)
    ts = [t + window_length // 2 for  t in ts]
    h = ax2.scatter(ts, azs, c=ls, cmap="magma")
    ax2.tick_params(axis='y', labelcolor="purple")
    ax2.set_ylabel("Azimuth", color="purple")
    plt.legend([h], ["Color: Linearity of motion"], loc=1)
    plt.xticks([])
    plt.xlim(xwin)
    plt.title("Wu (Time domain)")
    #print(azs)
    #plt.show()


    #fig = plt.figure()
    ax3 = fig.add_subplot(212)
    tr = tr_keep.copy()
    ts, azs, ps, ls, ams, pgvs = pol_win(tr, window_length, step=window_step, method="frequency", moving_average=moving_average)
    #ax3 = fig.add_subplot(212)
    azs = np.array(azs)
    t = np.linspace(0, tr[0].stats.npts * tr[0].stats.delta, tr[0].stats.npts)
    ax3.plot(t, tr[0].data, "k", alpha=0.5)
    ax3.plot(t, tr[1].data, "k", alpha=0.5)
    ax3.plot(t, tr[2].data, "k", alpha=0.5)
    #ax.tick_params(axis='y', labelcolor="r")
    plt.xlabel("Time (s)")
    ax3.set_ylabel("Seismic timeseries", color="k")
    ax4 = ax3.twinx()
    ts = [t + window_length // 2 for  t in ts]
    h = ax4.scatter(ts, azs, c=ls, cmap=plt.cm.magma)
    ax4.tick_params(axis='y', labelcolor="purple")
    ax4.set_ylabel("Azimuth", color="purple")
    plt.title("Vidale (using analytic signal)")
    plt.legend([h], ["Color: Strength of polarization"], loc=1)
    plt.xlim(xwin)
    plt.savefig("example_{}.png".format(example))
    plt.show()





    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(211)
    t = np.linspace(0, tr[0].stats.npts * tr[0].stats.delta, tr[0].stats.npts)
    ax.plot(t, tr[0].data, "k", alpha=0.5)
    ax.plot(t, tr[1].data, "k", alpha=0.5)
    ax.plot(t, tr[2].data, "k", alpha=0.5)
    ax.tick_params(axis='y', labelcolor="k")
    ax.set_ylabel("Seismic timeseries", color="k")
    tr = tr_keep.copy()
    ts, azs, ps, ls, ams, pgvs = pol_win(tr, window_length, step=window_step, moving_average=moving_average)
    ax2 = ax.twinx()
    azs = np.array(azs)
    ts = [t + window_length // 2 for  t in ts]
    h = ax2.scatter(ts, azs, c=ps, cmap="viridis")
    ax2.tick_params(axis='y', labelcolor="teal")
    ax2.set_ylabel("Azimuth", color="teal")
    plt.legend([h], ["Color: Planarity"], loc=1)
    plt.xticks([])
    plt.xlim(xwin)
    plt.title("Wu (Time domain)")
    #print(azs)
    #plt.show()


    #fig = plt.figure()
    ax3 = fig.add_subplot(212)
    tr = tr_keep.copy()
    ts, azs, ps, ls, ams, pgvs = pol_win(tr, window_length, step=window_step, method="frequency", moving_average=moving_average)
    #ax3 = fig.add_subplot(212)
    azs = np.array(azs)
    t = np.linspace(0, tr[0].stats.npts * tr[0].stats.delta, tr[0].stats.npts)
    ax3.plot(t, tr[0].data, "k", alpha=0.5)
    ax3.plot(t, tr[1].data, "k", alpha=0.5)
    ax3.plot(t, tr[2].data, "k", alpha=0.5)
    #ax.tick_params(axis='y', labelcolor="r")
    plt.xlabel("Time (s)")
    ax3.set_ylabel("Seismic timeseries", color="k")
    ax4 = ax3.twinx()
    ts = [t + window_length // 2 for  t in ts]
    h = ax4.scatter(ts, azs, c=ps, cmap=plt.cm.viridis)
    ax4.tick_params(axis='y', labelcolor="teal")
    ax4.set_ylabel("Azimuth", color="teal")
    plt.title("Vidale (using analytic signal)")
    plt.legend([h], ["Color: Ellipticity"], loc=1)
    plt.xlim(xwin)
    plt.savefig("example_{}2.png".format(example))
    plt.show()
    # fig = plt.figure()
    # ax = fig.add_subplot(211)
    # ax.plot(t, tr[0].data, "k", alpha=0.5)
    # ax.plot(t, tr[1].data, "k", alpha=0.5)
    # ax.plot(t, tr[2].data, "k", alpha=0.5)
    # ax.tick_params(axis='y', labelcolor="k")
    # ax.set_ylabel("Seismic timeseries", color="k")
    # ts, azs, ps, ls = pol_win(tr, window_length, step=window_step)
    # ax2 = ax.twinx()
    # ts = [t + window_length // 2 for  t in ts]
    # hh = ax2.scatter(ts, azs, c=ps, cmap=plt.cm.Greens_r)
    # ax2.tick_params(axis='y', labelcolor="g")
    # ax2.set_ylabel("Azimuth", color="g")
    # plt.legend([hh], ["Color: Planarity"], loc=1)
    # plt.xticks([])
    # plt.xlim(xwin)
    # plt.title("Wu (Time domain)")
    # #print(azs)
    # ts, azs, ps, ls = pol_win(tr, window_length, step=window_step, method="frequency")
    # ax = fig.add_subplot(212)
    # ax.plot(t, tr[0].data, "k", alpha=0.5)
    # ax.plot(t, tr[1].data, "k", alpha=0.5)
    # ax.plot(t, tr[2].data, "k", alpha=0.5)
    # #ax.tick_params(axis='y', labelcolor="r")
    # plt.xlabel("Time (s)")
    # ax.set_ylabel("Seismic timeseries", color="k")
    # ax2 = ax.twinx()
    # ts = [t + window_length // 2 for  t in ts]
    # hh1 = ax2.scatter(ts, azs, c=ps, cmap=plt.cm.Greens_r)
    # ax2.tick_params(axis='y', labelcolor="g")
    # ax2.set_ylabel("Azimuth", color="g")
    # plt.title("Vidale (using analytic signal)")
    # plt.legend([hh1], ["Color: Ellipticity"], loc=1)
    # plt.xlim(xwin)
    # plt.show()