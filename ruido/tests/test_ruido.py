# coding: utf-8
from ruido.classes.cc_dataset_mpi import CCDataset
import numpy as np
from ruido.utils.noisepy import stretching_vect
from obspy.signal.invsim import cosine_taper
import pytest

def test_stretching_vect():
    t = np.linspace(0., 10.0, 2500)
    original_signal = np.sin(t * 10.) * cosine_taper(2500, p=0.75)

    t_stretch = np.linspace(0., 9.95, 2500)  # 0.5 % perturbation
    stretched_signal = np.interp(t, t_stretch, original_signal)


    para = {}
    para["dt"] = 1. / 250.
    para["twin"] = [0., 10.]  # [self.lag[0], self.lag[-1] + 1. / self.fs]
    para["freq"] = [9.9, 10.1]

    dvv,  error, cc, cdp = stretching_vect(ref=original_signal, cur=stretched_signal,
                                      dv_range=0.05, nbtrial=100, para=para)
    assert pytest.approx(cc) == 1.0
    assert abs(dvv - 0.5) < para["dt"]   # verify that dv/v is 0.5%


def test_cc_dataset():
    testdata = CCDataset("tests/testdata/TO.MULU.01.HHZ--TO.MULU.01.HHN.ccc.windows.h5")

    s = testdata.__str__()
    assert s == "Cross-correlation dataset: TO.MULU.01.HHZ--TO.MULU.01.HHN\n"

    testdata.data_to_memory()
    assert testdata.dataset[0].ntraces == 10
    assert len(testdata.dataset) == 1

    testdata.stack([0, 1, 2, 3, 4], stacklevel_in=0, stacklevel_out=1)
    testdata.stack([5, 6, 7, 8, 9], stacklevel_in=0, stacklevel_out=1)
    assert testdata.dataset[1].ntraces == 2
    print(testdata.dataset[1].lag)

    testdata.add_datafile("tests/testdata/TO.MULU.01.HHZ--TO.MULU.01.HHN.ccc.windows.h5")
    testdata.data_to_memory(keep_duration=3600)
    assert testdata.dataset[0].ntraces == 17

def test_cc_data():
    testdata = CCDataset("tests/testdata/TO.MULU.01.HHZ--TO.MULU.01.HHN.ccc.windows.h5")
    testdata.data_to_memory()

    dat = testdata.dataset[0]

    dat.post_whiten(f1=0.1, f2=5.0)

    dat.remove_nan_segments()
    # once test data synthetics are done add test here

    dat.add_rms()
    # once test data synthetics are done add a test of expected value here

    clusters = np.zeros((2, dat.ntraces))
    clusters[0] = dat.timestamps
    clusters[0, 0] = 0
    clusters[1] = range(dat.ntraces)
    dat.add_cluster_labels(clusters)

    assert dat.cluster_labels[0] == -1
    assert dat.cluster_labels[1] == 1



