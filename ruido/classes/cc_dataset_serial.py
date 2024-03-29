import numpy as np
import h5py
from obspy import UTCDateTime
from obspy.signal.invsim import cosine_taper
import matplotlib.pyplot as plt
from scipy.signal import sosfilt, sosfiltfilt, hann, tukey, fftconvolve
from scipy.fftpack import next_fast_len
from scipy.interpolate import interp1d
from ruido.utils import filter
import os
from ruido.utils.noisepy import dtw_dvv, stretching_vect,\
    mwcs_dvv, robust_stack
from ruido.utils.whiten import whiten
# from ruido.clustering import cluster, cluster_minibatch
from obspy.signal.filter import envelope
from obspy.signal.detrend import polynomial as obspolynomial

"""
Module for handling auto- or cross-correlation datasets
as output by ants_2 python code.
"""

class CCData_serial(object):
    """
    An object that holds an array of timestamps, an array of correlation data,
    and the sampling rate. In addition it holds some helpful derived
    metadata (like nr. of traces).

    ...

    Attributes
    ----------
    :type data:  :class:`~numpy.ndarray`
    :param data: 2-D data array where row is observation time,
    and column is correlation lag
    :type timestamps: :class:`~numpy.ndarray`
    :param timestamps: Array of POSIX timestamps output by obspy via ants_2
    :type fs: float
    :param fs: Sampling rate in Hz

    Methods
    -------
    remove_nan_segments(self): Remove traces that contain numpy NaNs
    add_rms(self): Add an array containing the root mean square amplitude of
    each observation time
    add_cluster_labels(self, clusters): Append labels from clustering file,
    double checking consistent time stamps
    align(self, t1, t2, ref, plot=False): Shift traces to maximize the
    correlation coefficient of samples between lags t1 and t2,
    with respect to reference ref

    """

    def __init__(self, data, timestamps, fs):

        self.data = np.array(data)
        self.npts = self.data.shape[1]
        self.ntraces = self.data.shape[0]
        if self.ntraces == 1 and np.ndim(timestamps) == 0:
            self.timestamps = np.array([timestamps])
        else:
            self.timestamps = np.array(timestamps)
        self.fs = fs
        self.ntraces = self.data.shape[0]
        self.max_lag = (self.npts - 1) / 2 / self.fs
        self.lag = np.linspace(-self.max_lag, self.max_lag, self.npts)
        self.cluster_labels = None
        self.add_rms()
        self.median = np.nanmedian(self.data, axis=0)

    def remove_nan_segments(self):
        ntraces = self.data.shape[0]
        ixfinite = np.isfinite(self.data.sum(axis=1))
        if ixfinite.sum() == ntraces:
            return()

        self.data = self.data[ixfinite]
        ntraces_new = self.data.shape[0]
        print("Removed {} of {} traces due to NaN values.".format(ntraces -
            ntraces_new, ntraces))
        print("Dates of removed segments:")
        for t in self.timestamps[np.invert(ixfinite)]:
            print(UTCDateTime(t))

        self.timestamps = self.timestamps[ixfinite]
        self.ntraces = ntraces_new

    def add_rms(self):
        # add root mean square of raw cross-correlation windows
        # (for selection)
        rms = np.zeros(self.ntraces)

        for i, dat in enumerate(self.data):
            rms[i] = np.sqrt(((dat - np.nanmean(dat)) ** 2).mean())
            if np.isnan(rms[i]):
                print("Nan during RMS: ", rms[i], i, dat.mean())
                # make the value large so that these windows get discarded
                rms[i] = 1.0e4
        self.rms = rms

    def add_cluster_labels(self, clusters):

        if type(clusters) != np.ndarray:
            c = np.load(clusters)
        else:
            c = clusters
        cl = []
        for tst in self.timestamps:
            try:
                # this is slow but ensures consistency
                # only done once
                ix = np.where(c[0] == tst)[0][0]
                cl.append(int(c[1, ix]))
            except IndexError:
                cl.append(-1)
        cl = np.array(cl)
        print("Nr of unmatched timestamps: ", len(cl[cl == -1]))
        self.cluster_labels = cl

    def align(self, t1, t2, ref, plot=False):
        l0 = np.argmin((self.lag - t1) ** 2)
        l1 = np.argmin((self.lag - t2) ** 2)

        taper = np.ones(self.lag.shape)
        taper[0: l0] = 0
        taper[l1:] = 0
        taper[l0: l1] = tukey(l1-l0)
        opt_shifts = np.zeros(self.timestamps.shape)
        for i in range(len(opt_shifts)):
            test = self.data[i] * taper
            ix0 = len(ref) // 2
            ix1 = len(ref) // 2 + len(ref)
            cc = fftconvolve(test[::-1] / test.max(), ref / ref.max(), "full")
            cc = cc[ix0: ix1]
            shift = int(self.lag[np.argmax(cc)] * self.fs)

            # apply the shift
            if shift == 0:
                pass
            elif shift > 0:
                self.data[i, shift:] = self.data[i, : -shift].copy()
                self.data[i, 0: shift] = 0
            else:
                self.data[i, :shift] = self.data[i, -shift:].copy()
                self.data[i, shift:] = 0

    def data_to_envelope(self):
        # replace stacks by their envelope
        newstacks = []
        for s in self.data:
            newstacks.append(envelope(s))
        self.dataset = np.array(newstacks)

    def select_by_percentile(self, ixs, perc=90,
                             mode="upper", debug_mode=False):
        # select on the basis of relative root mean square amplitude
        # of the cross-correlations
        if self.rms is None:
            self.add_rms()
        rms = self.rms[ixs]
        if mode == "upper":
            ixs_keep = np.where(rms <= np.nanpercentile(rms, perc))
        elif mode == "lower":
            ixs_keep = np.where(rms >= np.nanpercentile(rms, perc))
        elif mode == "both":
            ixs_keep = np.intersect1d(np.where(rms <= np.nanpercentile(rms, perc)),
                                      np.where(rms >= np.nanpercentile(rms, 100 - perc)))
        if debug_mode:
            print("Selection by percentile of RMS: Before, After", len(ixs), len(ixs_keep))

        return(ixs[ixs_keep])

    def group_for_stacking(self, t0, duration, cluster_label=None):
        """
        Create a list of time stamps to be stacked
        --> figure out what "belongs together" in terms of time window
        afterwards, more elaborate selections can be applied :)
        Alternatively, group stacks together for forming longer stacks
        e.g. after clustering
        """

        t_to_select = self.timestamps

        # find closest to t0 window
        assert type(t0) in [float, np.float64, np.float32], "t0 must be floating point time stamp"

        # find indices
        ixs_selected = np.intersect1d(np.where(t_to_select >= t0),
                                      np.where(t_to_select < (t0 + duration)))

        # check if selection to do for clusters
        if cluster_label is not None:
            if self.cluster_labels is None:
                raise ValueError("Selection by cluster labels not possible: No labels assigned.")
            ixs_selected = np.intersect1d(ixs_selected,
                                          np.where(self.cluster_labels == cluster_label))

        print("grouped for stack: ", ixs_selected)
        return(ixs_selected)

    def select_for_stacking(self, ixs, selection_mode, cc=0.5,
                            twin=None, ref=None, dist=None, **kwargs):
        """
        Select by: closeness to reference or percentile or...
        Right now this only applies to raw correlations, not to stacks
        """
        lag = self.lag
        data = self.data

        if len(ixs) == 0:
            return([])

        if selection_mode == "rms_percentile":
            ixs_selected = self.select_by_percentile(ixs, **kwargs)

        elif selection_mode == "cc_to_median":
            median = self.median
            ixs_selected = []
            if twin is not None:
                cc_ixs = [np.argmin((lag - t) ** 2) for t in twin]
            else:
                cc_ixs = [0, -1]
            for i in ixs:
                corrcoeff = np.corrcoef(data[i, cc_ixs[0]: cc_ixs[1]], 
                                        median[cc_ixs[0]: cc_ixs[1]])[0, 1]
                if corrcoeff < cc or np.isnan(corrcoeff):
                    continue
                ixs_selected.append(i)
            ixs_selected = np.array(ixs_selected)

        elif selection_mode == "cc_to_ref":
            ixs_selected = []
            if ref is None:
                raise ValueError("Reference must be given.")
            if twin is not None:
                cc_ixs = [np.argmin((lag - t) ** 2) for t in twin]
            else:
                cc_ixs = [0, -1]
            for i in ixs:
                corrcoeff = np.corrcoef(data[i, cc_ixs[0]: cc_ixs[1]],
                                        ref[cc_ixs[0]: cc_ixs[1]])[0, 1]
                if corrcoeff < cc or np.isnan(corrcoeff):
                    continue
                ixs_selected.append(i)
            ixs_selected = np.array(ixs_selected)

        else:
            raise NotImplementedError

        print("selected for stack: ", ixs_selected)
        return(ixs_selected)

    def demean(self):
        to_demean = self.data

        for d in to_demean:
            d -= d.mean()

    def detrend(self, order=3):
        to_detrend = self.data
        for d in to_detrend:
            obspolynomial(d, order=order)

    def filter_data(self, taper_perc=0.1, filter_type="bandpass",
                    f_hp=None, f_lp=None, corners=4, zerophase=True,
                    maxorder=8):

        """
        Parallel filtering using scipy second order section filter
        """

        to_filter = self.data
        npts = self.npts
        fs = self.fs

        # define taper to avoid high-freq. artefacts
        taper = cosine_taper(npts, taper_perc)

        # define filter
        if filter_type == 'bandpass':
            if None in [f_hp, f_lp]:
                raise TypeError("f_hp and f_lp (highpass and lowpass frequency) must be floats.")
            sos = filter.bandpass(df=fs, freqmin=f_hp, freqmax=f_lp,
                                  corners=corners)
        elif filter_type == 'lowpass':
            sos = filter.lowpass(df=fs, freq=f_lp, corners=corners)
        elif filter_type == 'highpass':
            sos = filter.highpass(df=fs, freq=f_hp, corners=corners)
        elif filter_type == "cheby2_bandpass":
            sos = filter.cheby2_bandpass(df=fs, freq0=f_hp, freq1=f_lp,
                                         maxorder=maxorder)
        else:
            msg = 'Filter %s is not implemented, implemented filters:\
            bandpass, highpass, lowpass, cheby2_bandpass' % type
            raise ValueError(msg)

        for i, tr in enumerate(to_filter):
            if zerophase:
                to_filter[i, :] = sosfiltfilt(sos, taper * tr, padtype="even")
            else:
                to_filter[i, :] = sosfilt(sos, taper * tr)

    def window_data(self, t_mid, hw, window_type="tukey",
                    tukey_alpha=0.2, cutout=False):
        # check that the input array has 2 dimensions
        to_window = self.data
        lag = self.lag
        if not np.ndim(to_window) == 2:
            raise ValueError("Input array for windowing must have dimensions of n_traces * n_samples")

        win = filter.get_window(t_mid, hw, lag, window_type=window_type, alpha=tukey_alpha)

        if not cutout:
            for ix in range(to_window.shape[0]):
                to_window[ix, :] *= win
        else:
            new_win_dat = []
            for ix in range(to_window.shape[0]):
                ix_to_keep = np.where(win > 0.0)[0]
                new_win_dat.append(to_window[ix, ix_to_keep])
            newlag = self.lag[ix_to_keep]
            self.lag = newlag
            self.data = np.array(new_win_dat)
            self.npts = len(ix_to_keep)

    # def run_measurement(self, indices, to_measure, timestamps,
    #                     ref, fs, lag, f0, f1, ngrid,
    #                     dvv_bound, method="stretching"):
        
    #     reference = ref.copy()
    #     para = {}
    #     para["dt"] = 1. / fs
    #     para["twin"] = [lag[0], lag[-1] + 1. / fs]
    #     para["freq"] = [f0, f1]

    #     if indices is None:
    #         indices = range(len(to_measure))

    #     dvv_times = np.zeros(len(indices))
    #     ccoeff = np.zeros(len(indices))
    #     best_ccoeff = np.zeros(len(indices))

    #     if method in ["stretching", "mwcs"]:
    #         dvv = np.zeros((len(indices), 1))
    #         dvv_error = np.zeros((len(indices), 1))
            
    #     elif method in ["dtw"]:
    #         if len_dtw_msr is None:
    #             len_dtw_msr = []
    #             testmsr = dtw_dvv(reference, reference,
    #                           para, maxLag=maxlag_dtw,
    #                           b=10, direction=1)
    #             len_dtw_msr.append(len(testmsr[0]))
    #             len_dtw_msr.append(testmsr[1].shape)

    #         dvv = np.zeros((len(indices), len_dtw_msr[0]))
    #         dvv_error = np.zeros((len(indices), *len_dtw_msr[1]))
    #     else:
    #         raise ValueError("Unknown measurement method {}.".format(method))

    #     for i, tr in enumerate(to_measure):
    #         if i not in indices:
    #             dvv[i, :] = np.nan
    #             dvv_times[i] = timestamps[i]
    #             ccoeff[i] = np.nan
    #             # print(ccoeff[cnt])
    #             best_ccoeff[i] = np.nan
    #             dvv_error[i, :] = np.nan
    #             continue

    #         if method == "stretching":
    #             dvvp, delta_dvvp, coeffp, cdpp = stretching_vect(reference, tr,
    #                                                     dvv_bound, ngrid, para)
    #         elif method == "dtw":
    #             dvv_bound = int(dvv_bound)
    #             warppath, dist,  coeffor, coeffshift = dtw_dvv(reference, tr,
    #                                      para, maxLag=maxlag_dtw,
    #                                      b=dvv_bound, direction=1)
    #             coeffp = coeffshift
    #             cdpp = coeffor
    #             delta_dvvp = dist
    #             dvvp = warppath
    #         elif method == "mwcs":
    #             ixsnonzero = np.where(reference != 0.0)
    #             dvvp, errp = mwcs_dvv(reference[ixsnonzero],
    #                                  tr[ixsnonzero],
    #                                  moving_window_length,
    #                                  moving_window_step, para)
    #             delta_dvvp = errp
    #             coeffp = np.nan
    #             cdpp = np.nan

    #         dvv[i, :] = dvvp
    #         dvv_times[i] = timestamps[i]
    #         ccoeff[i] = cdpp
    #         best_ccoeff[i] = coeffp
    #         dvv_error[i, :] = delta_dvvp
    #     return(dvv, dvv_times, ccoeff, best_ccoeff, dvv_error)

    def measure_dvv_ser(self, ref, f0, f1, method="stretching",
                        ngrid=90, dvv_bound=0.03,
                        measure_smoothed=False, indices=None,
                        moving_window_length=None, moving_window_step=None,
                        maxlag_dtw=0.0,
                        len_dtw_msr=None):

        if indices is None:
            indices = np.arange(len(self.data))

        to_measure = self.data[indices]
        lag = self.lag
        timestamps = self.timestamps[indices]
        # print(timestamps)
        fs = self.fs

        if len(to_measure) == 0:
            return()

        reference = ref.copy()
        para = {}
        para["dt"] = 1. / fs
        para["twin"] = [lag[0], lag[-1] + 1. / fs]
        para["freq"] = [f0, f1]

        dvv_times = np.zeros(len(indices))
        ccoeff = np.zeros(len(indices))
        best_ccoeff = np.zeros(len(indices))

        if method in ["stretching", "mwcs"]:
            dvv = np.zeros((len(indices), 1))
            dvv_error = np.zeros((len(indices), 1))
        elif method in ["dtw"]:
            if len_dtw_msr is None:
                len_dtw_msr = []
                testmsr = dtw_dvv(reference, reference,
                                  para, maxLag=maxlag_dtw,
                                  b=10, direction=1)
                len_dtw_msr.append(len(testmsr[0]))
                len_dtw_msr.append(testmsr[1].shape)

            dvv = np.zeros((len(indices), len_dtw_msr[0]))
            dvv_error = np.zeros((len(indices), *len_dtw_msr[1]))
        else:
            raise ValueError("Unknown measurement method {}.".format(method))

        for i, tr in enumerate(to_measure):

            if np.any(np.isnan(tr)):
                dvv[i, :] = np.nan
                dvv_times[i] = timestamps[i]
                ccoeff[i] = np.nan
                best_ccoeff[i] = np.nan
                dvv_error[i, :] = np.nan
                continue
            if method == "stretching":
                dvvp, delta_dvvp, coeffp, cdpp = stretching_vect(reference, tr,
                                                                 dvv_bound,
                                                                 ngrid, para)
            elif method == "dtw":
                dvv_bound = int(dvv_bound)
                warppath, dist,  \
                    coeffor, coeffshift = dtw_dvv(reference, tr,
                                                  para, maxLag=maxlag_dtw,
                                                  b=dvv_bound, direction=1)
                coeffp = coeffshift
                cdpp = coeffor
                delta_dvvp = dist
                dvvp = warppath
            elif method == "mwcs":
                ixsnonzero = np.where(reference != 0.0)
                dvvp, errp = mwcs_dvv(reference[ixsnonzero],
                                      tr[ixsnonzero],
                                      moving_window_length,
                                      moving_window_step, para)
                delta_dvvp = errp
                coeffp = np.nan
                cdpp = np.nan

            dvv[i, :] = dvvp
            dvv_times[i] = timestamps[i]
            ccoeff[i] = cdpp
            best_ccoeff[i] = coeffp
            dvv_error[i, :] = delta_dvvp
        return(dvv, dvv_times, ccoeff, best_ccoeff, dvv_error)


    def post_whiten(self, f1, f2, npts_smooth=5):

        # nfft = int(next_fast_len(self.npts))
        td_taper = cosine_taper(self.npts, 0.1)
        to_filter = self.data
        npts = self.npts
        # print(npts)
        fs = self.fs

        for i, tr in enumerate(to_filter):
            spec, nfft = whiten(td_taper * tr, fs, f1, f2,
                                n_smooth=npts_smooth)
            to_filter[i, :] = np.fft.irfft(spec, n=nfft)[0: npts]


    def interpolate_stacks(self, new_fs):
        """
        Cubic interpolation to new sampling rate
        More fancy interpolations (like Lanczos) are a bit too expensive 
        """
        if (new_fs % self.fs) != 0:
            raise ValueError("Only integer-factor resampling is permitted.")

        stacks = self.data
        new_npts = int((self.npts - 1.) / self.fs * new_fs) + 1
        new_lag = np.linspace(-self.max_lag, self.max_lag, new_npts)

        newstacks = []
        for stack in stacks:
            f = interp1d(self.lag, stack, kind="cubic")
            newstacks.append(f(new_lag))
        self.data = np.array(newstacks)
        self.lag = new_lag
        self.npts = new_npts
        self.fs = 1. / (new_lag[1] - new_lag[0])


class CCDataset_serial(object):
    """
    An object that holds a dictionary of CCData objects and can perform all sorts
    of stuff on them. For all operations like 
    ...

    Attributes
    ----------
    :type inputfile:  string
    :param inpytfile: File path to hdf5 file produced by ants_2 python code
    :type ref: numpy.ndarray or None
    :param ref: reference trace (can be passed later)
    

    Methods
    -------
    add_datafile(self, inputfile)
    __str__(self)
    data_to_memory(self,ix_corr_max=None, ix_corr_min=None, keep_duration=0,
    normalize=False)
    stack(self, ixs, stackmode="linear", stacklevel_in=0, stacklevel_out=1, overwrite=False,
              epsilon_robuststack=None)
    """

    def __init__(self, inputfile):
        """
        :type inputfile: string
        :param inputfile: File path to hdf5 file
        :type ref: numpy.ndarray or None
        :param ref: reference trace, e.g. from another file

        """

        self.station_pair = os.path.splitext(os.path.basename(inputfile))[0]
        self.station_pair = os.path.splitext(self.station_pair)[0]
        self.station_pair = os.path.splitext(self.station_pair)[0]
        self.datafile = h5py.File(inputfile, 'r')
        self.dataset = {}

    def add_datafile(self, inputfile):

        self.datafile.close()
        self.station_pair = os.path.splitext(os.path.basename(inputfile))[0]
        self.station_pair = os.path.splitext(self.station_pair)[0]
        self.station_pair = os.path.splitext(self.station_pair)[0]
        self.datafile = h5py.File(inputfile, 'r')

    def __str__(self):

        output = ""
        output += "Cross-correlation dataset: {}\n".format(self.station_pair)
        for (k, v) in self.dataset.items():
            output += "Contains {} traces on stacking level {}\n".format(v.ntraces, k)
            output += "Starting {}, ending {}\n".format(UTCDateTime(v.timestamps[0]).strftime("%d.%m.%Y"),
                                                        UTCDateTime(v.timestamps[-1]).strftime("%d.%m.%Y"))
        return(output)


    def data_to_memory(self, ix_corr_max=None, ix_corr_min=None, keep_duration=0,
                       normalize=False):

        """
        Read data into memory and store in dataset[0]. Serial version (needed for clustering).
        :type ix_corr_max: int
        :param ix_corr_max: correlation traces will be read up unto the preceding index
        :type ix_corr_min: int
        :param ix_corr_min: correlation traces will be read starting at this index
        :type keep_duration: int
        :param keep duration: nr. of correlation traces to keep in dataset counting from the
        end of the array (i.e. later in time).
        The preceding traces will be discarded from memory.
        :type normalize: Boolean
        :param normalize: normalize each trace by its maximum or not
        """
        # read data from correlation file into memory and store in dataset[0]
        # use self.datafile
        # get fs, data, timestamps
        # commit to a new dataset object or add it to existing
        fs = dict(self.datafile['stats'].attrs)['sampling_rate']
        npts = self.datafile['corr_windows']["data"].shape[1]
        ntraces = len(self.datafile["corr_windows"]["timestamps"])

        if ix_corr_max is None:
            ix_corr_max = ntraces
        if ix_corr_min is None:
            ix_corr_min = 0

        n_to_read = ix_corr_max - ix_corr_min

        # allocate data
        try:
            data = np.zeros((ix_corr_max - ix_corr_min, npts))
        except MemoryError:
            print("Data doesn't fit in memory, set a lower n_corr_max")
            return()

        # allocate timestamps array
        timestamps = np.zeros(ix_corr_max - ix_corr_min)
        for i in range(ix_corr_min, ix_corr_max):  # , v in enumerate(self.datafile["corr_windows"]["data"][:]):
            v = self.datafile["corr_windows"]["data"][i]
            tstamp = self.datafile["corr_windows"]["timestamps"][i]
            data[i - ix_corr_min, :] = v
            if tstamp == "":
                # leave timestamp as 0, will get dropped
                continue
            if type(tstamp) in [np.float64, np.float32, float]:
                timestamps[i - ix_corr_min] = tstamp
            else:
                try:
                    #print(str(tstamp), type(str(tstamp)))
                    #print(str(tstamp).split(".")[0:5])
                    #tstmp = b",".join([tstt for tstt in tstamp.split(b".")[0: 5]])
                    ttstmp = str(tstamp).split(".")
                    tstmp = "{},{},{},{},{}".format(*ttstmp[0: 5])
                    print(tstmp)
                    print(type(tstmp.encode()), type(b"2021,032,00,00,00"))
                    print(tstmp.encode("utf-8") == b'2021,032,00,00,00')
                    timestamps[i - ix_corr_min] = UTCDateTime(tstmp).timestamp
                except KeyError:
                    # leave timestamp as 0, will get dropped
                    pass

        data = data[timestamps != 0.0]
        timestamps = timestamps[timestamps != 0.0]

        if normalize:
            # absolute last-resort-I-cannot-reprocess way to deal with amplitude issues
            for tr in data:
                tr /= tr.max()


        if len(timestamps) > 0:

            try:
                if keep_duration != 0:
                    if keep_duration > 0:
                        ixcut = np.argmin(((self.dataset[0].timestamps[-1] -
                                            self.dataset[0].timestamps) -
                                           keep_duration) ** 2)

                    else:  # keep all if negative keep_duration
                        ixcut = 0
                    self.dataset[0].data = np.concatenate((self.dataset[0].data[ixcut:], np.array(data, ndmin=2)), axis=0)
                    self.dataset[0].timestamps = np.concatenate((self.dataset[0].timestamps[ixcut:], timestamps), axis=None)
                else:
                    self.dataset[0].data = np.array(data, ndmin=2)
                    self.dataset[0].timestamps = timestamps
            except KeyError:
                self.dataset[0] = CCData_serial(np.array(data, ndmin=2), timestamps, fs)

            self.dataset[0].ntraces = self.dataset[0].data.shape[0]
            self.dataset[0].npts = self.dataset[0].data.shape[1]
            self.dataset[0].add_rms()
            self.dataset[0].remove_nan_segments()
            self.dataset[0].median = np.nanmedian(self.dataset[0].data, axis=0)
            # sort -- ants sometimes has weird little back jumps to deal with gaps
            ixs_time = np.argsort(self.dataset[0].timestamps)
            self.dataset[0].data = self.dataset[0].data[ixs_time, :]
            self.dataset[0].timestamps = self.dataset[0].timestamps[ixs_time]
            print("Read to memory from {} to {}".format(UTCDateTime(timestamps[0]),
                                                        UTCDateTime(timestamps[-1])))
        else:
            self.dataset = {}


    def stack(self, ixs, stackmode="linear", stacklevel_in=0, stacklevel_out=1,
              epsilon_robuststack=None):
        #stack
        if len(ixs) == 0:
            return()

        to_stack = self.dataset[stacklevel_in].data
        t_to_stack = self.dataset[stacklevel_in].timestamps

        if stackmode == "linear":
            s = to_stack[ixs].sum(axis=0).copy()
            newstacks = s / len(ixs)
            newt = t_to_stack[ixs[0]]
        elif stackmode == "median":
            newstacks = np.median(to_stack[ixs], axis=0)
            newt = t_to_stack[ixs[0]]
        elif stackmode == "robust":
            newstacks, w, nstep = robust_stack(to_stack[ixs], epsilon_robuststack)
            print(newstacks.shape, " NEWSTACKS ", nstep)
            newt = t_to_stack[ixs[0]]
        else:
            raise ValueError("Unknown stacking mode {}".format(stackmode))
        
        try:
            self.dataset[stacklevel_out].data = np.concatenate((self.dataset[stacklevel_out].data, np.array(newstacks, ndmin=2)), axis=0)
            self.dataset[stacklevel_out].timestamps = np.concatenate((self.dataset[stacklevel_out].timestamps, newt), axis=None)

        except KeyError:
            self.dataset[stacklevel_out] = CCData_serial(np.array(newstacks, ndmin=2), newt, self.dataset[stacklevel_in].fs)
        
        self.dataset[stacklevel_out].ntraces = self.dataset[stacklevel_out].data.shape[0]
        self.dataset[stacklevel_out].npts = self.dataset[stacklevel_out].data.shape[1]

    def plot_stacks(self, stacklevel=1, outfile=None, seconds_to_show=20, scale_factor_plotting=None,
                    plot_mode="heatmap", seconds_to_start=0.0, cmap=plt.cm.bone, absolute_v=None,
                    mask_gaps=False, step=None, figsize=None,
                    color_by_cc=False, normalize_all=False, label_style="month",
                    ax=None, plot_envelope=False, ref=None, color_by_clusterlabel=False, each_n_labels=1,
                    mark_events=[], grid=True, marklags=[], colorful_traces=False, utcoffset=0.0):
        if mask_gaps and step == None:
            raise ValueError("To mask the gaps, you must provide the step between successive windows.")

        to_plot = self.dataset[stacklevel].data
        t_to_plot = self.dataset[stacklevel].timestamps
        lag = self.dataset[stacklevel].lag

        if plot_mode == "heatmap":
            # center on lags
            lag += 1. / self.dataset[stacklevel].fs

        if to_plot.shape[0] == 0:
            return()

        ylabels = []
        ylabelticks = []
        months = []
        years = []
        jdays = []

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax1 = fig.add_subplot(111)
        else:
            ax1 = ax

        if plot_mode == "traces":
            cnt = 0
            for i, tr in enumerate(to_plot):

                if color_by_cc:
                    cmap = plt.cm.get_cmap("Spectral", 20)
                    crange = np.linspace(-1.0, 1.0, 20)
                    cmap = cmap(crange)
                    cccol = np.corrcoef(tr, ref)[0][1]
                    ix_colr = np.argmin((cccol - crange) ** 2)
                    ax1.plot(lag, tr / tr.max() + scale_factor_plotting * cnt,
                             c=cmap[ix_colr], alpha=0.5)
                else:
                    if colorful_traces:
                        cmap = plt.cm.get_cmap("Spectral", self.dataset[stacklevel].ntraces)
                        cmap = cmap(range(self.dataset[stacklevel].ntraces))
                        ax1.plot(lag, tr / tr.max() + scale_factor_plotting * cnt,
                                linewidth=2., color=cmap[i])
                    else:
                        ax1.plot(lag, tr / tr.max() + scale_factor_plotting * cnt,
                                 'k', alpha=0.5, linewidth=0.5)

                t = t_to_plot[i]
                if label_style == "hour":
                    if UTCDateTime(t).strftime("%m.%d-%I%p") not in months:
                        ylabels.append(scale_factor_plotting * cnt)
                        ylabelticks.append(UTCDateTime(t + utcoffset).strftime("%I%p"))
                        months.append(UTCDateTime(t).strftime("%m.%d-%I%p"))
                if label_style == "month":
                    if UTCDateTime(t).strftime("%Y%m") not in months:
                        ylabels.append(scale_factor_plotting * cnt)
                        ylabelticks.append(UTCDateTime(t).strftime("%Y/%m/%d"))
                        months.append(UTCDateTime(t).strftime("%Y%m"))
                elif label_style == "year":
                    if UTCDateTime(t).strftime("%Y") not in years:
                        ylabelticks.append(UTCDateTime(t).strftime("%Y./%m/%d"))
                        ylabels.append(scale_factor_plotting * cnt)
                        years.append(UTCDateTime(t).strftime("%Y"))
                elif label_style == "jday":
                    if UTCDateTime(t).strftime("%Y%j") not in jdays:
                        ylabelticks.append(UTCDateTime(t).strftime("%Y.%j"))
                        ylabels.append(scale_factor_plotting * cnt)
                        jdays.append(UTCDateTime(t).strftime("%Y%j"))
                cnt += 1

            if ref is not None:
                ax1.plot(lag, ref / ref.max(), linewidth = 2.0)
                ax1.set_ylim([-1. - scale_factor_plotting,
                             scale_factor_plotting * cnt + np.mean(abs(tr))])
            else:
                ax1.set_ylim([0 - np.mean(abs(tr)), 
                             scale_factor_plotting * cnt + np.mean(abs(tr))])

        elif plot_mode == "heatmap":

            if not mask_gaps:
                dat_mat = np.zeros((self.dataset[stacklevel].ntraces, self.dataset[stacklevel].npts))
                cluster_labels_plotting = np.ones(dat_mat.shape[0], dtype=int) * -1
                for ix, tr in enumerate(to_plot):
                    if normalize_all:
                        dat_mat[ix, :] = tr / tr.max()
                    else:
                        dat_mat[ix, :] = tr
                    if plot_envelope:
                        dat_mat[ix, :] = envelope(dat_mat[ix, :])
                    t = t_to_plot[ix]
                    if label_style == "hour":
                        if UTCDateTime(t).strftime("%m.%d-%I%p") not in months:
                            ylabels.append(t)
                            ylabelticks.append(UTCDateTime(t + utcoffset).strftime("%I%p"))
                            months.append(UTCDateTime(t).strftime("%m.%d-%I%p"))
                    if label_style == "month":
                        if UTCDateTime(t).strftime("%Y%m") not in months:
                            ylabels.append(t)
                            ylabelticks.append(UTCDateTime(t).strftime("%Y/%m/%d"))
                            months.append(UTCDateTime(t).strftime("%Y%m"))

                    elif label_style == "year":
                        if UTCDateTime(t).strftime("%Y") not in years:
                            ylabels.append(t)
                            ylabelticks.append(UTCDateTime(t).strftime("%Y/%m/%d"))
                            years.append(UTCDateTime(t).strftime("%Y"))
                    
                    elif label_style == "jday":
                        if UTCDateTime(t).strftime("%Y%j") not in jdays:
                            ylabelticks.append(UTCDateTime(t).strftime("%Y.%j"))
                            ylabels.append(t)
                            jdays.append(UTCDateTime(t).strftime("%Y%j"))

                        
                if plot_envelope:
                    if scale_factor_plotting is not None:
                        vmin = 0
                        vmax = scale_factor_plotting * dat_mat.max()
                    else:
                        vmin = 0
                        vmax = absolute_v
                else:
                    if scale_factor_plotting is not None:
                        vmin = -scale_factor_plotting * dat_mat.max()
                        vmax = scale_factor_plotting * dat_mat.max()
                    else:
                        vmin = -absolute_v
                        vmax = absolute_v

            else:
                tstamp0 = t_to_plot[0]
                tstamp1 = t_to_plot[-1]
                t_to_plot_all = np.arange(tstamp0, tstamp1 + step, step=step)
                dat_mat = np.zeros((len(t_to_plot_all), self.dataset[stacklevel].npts))
                dat_mat[:, :] = np.nan
                cluster_labels_plotting = np.ones(dat_mat.shape[0], dtype=int) * -1
        
                for ix, tr in enumerate(to_plot):
                    t = t_to_plot[ix]
                    ix_t = np.argmin(np.abs(t_to_plot_all - t))
                    if color_by_clusterlabel:
                        cluster_labels_plotting[ix_t] = self.dataset[stacklevel].cluster_labels[ix]
                    if normalize_all:
                        dat_mat[ix_t, :] = tr / tr.max()
                    else:
                        dat_mat[ix_t, :] = tr
                    if plot_envelope:
                        dat_mat[ix_t, :] = envelope(dat_mat[ix_t, :])
                    if label_style == "hour":
                        if UTCDateTime(t).strftime("%m.%d-%I%p") not in months:
                            ylabels.append(t_to_plot_all[ix_t])
                            ylabelticks.append(UTCDateTime(t + utcoffset).strftime("%I%p"))
                            months.append(UTCDateTime(t).strftime("%m.%d-%I%p"))
                    if label_style == "month":
                        if UTCDateTime(t).strftime("%Y%m") not in months:
                            ylabels.append(t_to_plot_all[ix_t])
                            ylabelticks.append(UTCDateTime(t).strftime("%Y/%m/%d"))
                            months.append(UTCDateTime(t).strftime("%Y%m"))

                    elif label_style == "year":
                        if UTCDateTime(t).strftime("%Y") not in years:
                            ylabels.append(t_to_plot_all[ix_t])
                            ylabelticks.append(UTCDateTime(t).strftime("%Y/%m/%d"))
                            years.append(UTCDateTime(t).strftime("%Y"))
                    elif label_style == "jday":
                        if UTCDateTime(t).strftime("%Y%j") not in jdays:
                            ylabelticks.append(UTCDateTime(t).strftime("%Y.%j"))
                            ylabels.append(t_to_plot_all[ix_t])
                            jdays.append(UTCDateTime(t).strftime("%Y%j"))
                if plot_envelope:
                    vmin = 0
                    if scale_factor_plotting is not None:
                        vmax = scale_factor_plotting * np.nanmax(dat_mat)
                    else:
                        vmax = absolute_v

                else:
                    if scale_factor_plotting is not None:
                        vmin = -scale_factor_plotting * np.nanmax(dat_mat)
                        vmax = scale_factor_plotting * np.nanmax(dat_mat)
                    else:
                        vmin = -absolute_v
                        vmax = absolute_v
                t_to_plot = t_to_plot_all
            
           
            ax1.pcolormesh(lag, t_to_plot, dat_mat, vmax=vmax, vmin=vmin,
                            cmap=cmap)
            if color_by_clusterlabel:
                ax1.scatter(np.ones(len(t_to_plot))[::each_n_labels] * 0.95 * seconds_to_start, t_to_plot[::each_n_labels], c=cluster_labels_plotting[::each_n_labels], marker=".", cmap=plt.cm.rainbow)
                ax1.scatter(np.ones(len(t_to_plot)) * 0.95 * seconds_to_show, t_to_plot, c=cluster_labels_plotting, marker=".", cmap=plt.cm.rainbow)
                

        #if mark_17_quake:
        #    ylabels.append(UTCDateTime("2017,262").timestamp)
        #    ylabelticks.append("EQ Puebla")
        for m_ev in mark_events:
            ax1.hlines(xmin=seconds_to_start, xmax=seconds_to_show, y=UTCDateTime(m_ev).timestamp, linestyles="--", color="c")

        ax1.set_title(self.station_pair)
        if normalize_all:
            isnormalized = "Normalized"
        else:
            isnormalized = "Non-norm."
        ax1.set_ylabel("{} stacks / correlations (-)".format(isnormalized))
        ax1.set_xlim([seconds_to_start, seconds_to_show])
        ax1.set_xlabel("Lag (seconds)")
        ax1.set_yticks(ylabels)
        ax1.set_yticklabels(ylabelticks)
        ax1.yaxis.tick_right()

        for marklag in marklags:
            plt.plot([marklag, marklag], [t_to_plot.min(), t_to_plot.max()], "--", color="b")

        if grid:
            if not colorful_traces:
                ax1.grid(linestyle=":", color="lawngreen", axis="x")
            else:
                ax1.grid(axis="x")

        if seconds_to_show - seconds_to_start > 50:
            tickstep = 10.0
        elif seconds_to_show - seconds_to_start > 20:
            tickstep = 5.0
        elif seconds_to_show - seconds_to_start > 10:
            tickstep = 2.0
        else:
            tickstep = 1.0
        ax1.set_xticks([i for i in np.arange(seconds_to_start, seconds_to_show, tickstep)],
                       [str(i) for i in np.arange(seconds_to_start, seconds_to_show, tickstep)])

        if ax is None:
            if outfile is not None:
                plt.tight_layout()
                plt.savefig(outfile)
                plt.close()
            else:
                plt.show()
        else:
            return(ax1, t_to_plot)
