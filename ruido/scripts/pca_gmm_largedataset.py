# -*- coding: utf-8 -*-
from ruido.utils.Function_Clustering_DFs import run_pca, gmm
import numpy as np
from ruido.classes.cc_dataset_mpi import CCDataset, CCData  
import os
from glob import glob
from obspy import UTCDateTime
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# read input

for ixsta, station in enumerate(stations):
    ch1 = ch1s[ixsta]
    ch2 = ch2s[ixsta]

    datafiles = glob(os.path.join(input_directory,
                                  "*{}*{}*{}.*.h5".format(station,
                                                          ch1,
                                                          ch2)))
    datafiles.sort()
    print(datafiles)

    dset = CCDataset(datafiles[0])
    dset.data_to_memory()

    # select a random subset of traces for PCA
    # here we use dataset key 0 for the raw data read in from each file
    # to retain all the randomly selected windows, we copy them to key 1
    if type(n_samples_each_file) == int:
        ixs_random = np.random.choice(np.arange(dset.dataset[0].ntraces),
                                      min(n_samples_each_file,
                                      dset.dataset[0].traces))
    elif n_samples_each_file == "all":
        ixs_random = range(dset.dataset[0].ntraces)
    dset.dataset[1] = CCData(dset.dataset[0].data[ixs_random].copy(),
                             dset.dataset[0].timestamps[ixs_random].copy(),
                             dset.dataset[0].fs)

    for ixfile, dfile in enumerate(datafiles):
        if ixfile == 0:
            # we've been here already
            continue
        # read the data in
        dset.add_datafile(dfile)
        dset.data_to_memory(keep_duration=0)
        # use a random subset of each file (unless "all" requested)
        if type(n_samples_each_file) == int:
            ixs_random = np.random.choice(np.arange(dset.dataset[0].ntraces),
                                          min(n_samples_each_file,
                                          dset.dataset[0].traces))
        elif n_samples_each_file == "all":
            ixs_random = range(dset.dataset[0].ntraces)
        newdata = dset.dataset[0].data[ixs_random, :]
        assert (newdata.base is not dset.dataset[0].data)

        # keep the randomly selected windows under key 1, adding to the previously selected ones
        dset.dataset[1].data = np.concatenate((dset.dataset[1].data,
                                              newdata))
        dset.dataset[1].timestamps = np.concatenate((dset.dataset[1].timestamps,
                                                     dset.dataset[0].timestamps[ixs_random]))
        dset.dataset[1].ntraces = dset.dataset[1].data.shape[0]


    for fmin in fmins:
        # The clustering is performed separately in different frequency bands. The selections may change depending on the frequency band.
        fmax = 2 * fmin
        twin_hw = 10. / fmin
        # filter before clustering
        dset.filter_data(stacklevel=1, filter_type=filt_type,
                         f_hp=fmin, f_lp=fmax, maxorder=filt_maxord)
        #window. The windows are all centered on lag 0 and extend to 10 / fmin
        dset.window_data(t_mid=twin_mid, hw=twin_hw,
                         window_type="tukey", tukey_alpha=0.5,
                         stacklevel=1, cutout=False)

        # perform PCA on the random subset
        dset.dataset[1].data = np.nan_to_num(dset.dataset[1].data)
        X = StandardScaler().fit_transform(dset.dataset[1].data)
        pca_rand = run_pca(X, min_cumul_var_perc=expl_var)
        # pca output is a scikit learn PCA object
        # just for testing, run the Gaussian mixture here
        # gm = gmm(pca_rand.transform(X), range(1, 12))

        all_pccs = []
        all_timestamps = []

        # now go through all files again, read, filter, window, and fit the pcs
        for datafile in datafiles:
            print(datafile)
            dset.add_datafile(datafile)
            dset.data_to_memory(keep_duration=0)
            dset.filter_data(stacklevel=0, filter_type=filt_type,
                             f_hp=fmin, f_lp=fmax, maxorder=filt_maxord)
            dset.window_data(t_mid=twin_mid, hw=twin_hw,
                             window_type="tukey", tukey_alpha=0.5,
                             stacklevel=0, cutout=False)
            dset.dataset[0].data = np.nan_to_num(dset.dataset[0].data)
            X = StandardScaler().fit_transform(dset.dataset[0].data)
            # expand the data in the principal component basis:
            pca_output = pca_rand.transform(X)
            # append to the list
            all_pccs.extend(pca_output)
            all_timestamps.extend(dset.dataset[0].timestamps)
        all_pccs = np.array(all_pccs)
        all_timestamps = np.array(all_timestamps)

        # do the clustering
        gmmodels, n_clusters, gmixfinPCA, probs, BICF = gmm(all_pccs, nclust)
        print(n_clusters, np.unique(gmixfinPCA))
        # save the cluster labels
        labels = np.zeros((2, len(all_timestamps)))
        labels[0] = all_timestamps
        labels[1] = gmixfinPCA
        outputfile = "{}.{}.{}.{}-{}Hz.gmmlabels.npy".format(station, ch1, ch2, fmin, fmax)
        np.save(outputfile, labels)

