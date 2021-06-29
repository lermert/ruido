# -*- coding: utf-8 -*-
from ruido.utils.Function_Clustering_DFs import run_pca, gmm
import numpy as np
from ruido.classes.cc_dataset_serial import CCDataset_serial, CCData_serial
import os
from glob import glob
from obspy import UTCDateTime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# read input

def run_clustering(config, rank, size, comm):
    if rank == 0:
        print("*"*80)
        print("Running clustering.")
        print("*"*80)

    # if rank >= len(config["stations"]):
    #     if config["print_debug"]:
    #         print("Rank {} has nothing to do, exiting.".format(rank))
    #     return()

    # loop over stations
    ids_done = []
    for station1 in config["stations"][rank::size]:
        for station2 in config["stations"][rank::size]:
            if config["print_debug"]:
                print("Rank {} working on stations {}, {}.".format(rank, station1, station2))

            # loop over components
            for ixch1, ch1 in enumerate(config["channels"]):
                for ch2 in config["channels"][ixch1: ]:
                    channel_id = "{}.{}-{}.{}".format(station1, ch1, station2, ch2)
                    if channel_id in ids_done:
                        continue
                    else:
                        ids_done.append(channel_id)
                        if config["print_debug"]:
                            print("Rank {} clustering ".format(rank), channel_id)
                    if config["drop_autocorrelations"] and ch1 == ch2:
                        continue

                    datafiles = glob(os.path.join(config["input_directories"],
                                                  "*{}*{}*{}*{}.*windows.h5".format(station1,
                                                                          ch1,
                                                                          station2,
                                                                          ch2)))
                    if len(datafiles) == 0:
                        continue
                    datafiles.sort()

                    if config["print_debug"]:
                        print(datafiles)

                    dset = CCDataset_serial(datafiles[0])
                    dset.data_to_memory()

                    # select a random subset of traces for PCA
                    # here we use dataset key 0 for the raw data read in from each file
                    # to retain all the randomly selected windows, we copy them to key 1
                    if type(config["n_samples_each_file"]) == int:
                        ixs_random = np.random.choice(np.arange(dset.dataset[0].ntraces),
                                                      min(config["n_samples_each_file"],
                                                      dset.dataset[0].ntraces))
                    elif config["n_samples_each_file"] == "all":
                        ixs_random = range(dset.dataset[0].ntraces)
                    dset.dataset[1] = CCData_serial(dset.dataset[0].data[ixs_random].copy(),
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
                        if type(config["n_samples_each_file"]) == int:
                            ixs_random = np.random.choice(np.arange(dset.dataset[0].ntraces),
                                                          min(config["n_samples_each_file"],
                                                          dset.dataset[0].ntraces))
                        elif config["n_samples_each_file"] == "all":
                            ixs_random = range(dset.dataset[0].ntraces)

                        # create a new array
                        newdata = dset.dataset[0].data[ixs_random, :]
                        assert (newdata.base is not dset.dataset[0].data)

                        # keep the randomly selected windows under key 1, adding to the previously selected ones
                        dset.dataset[1].data = np.concatenate((dset.dataset[1].data,
                                                              newdata))
                        dset.dataset[1].timestamps = np.concatenate((dset.dataset[1].timestamps,
                                                                     dset.dataset[0].timestamps[ixs_random]))
                        dset.dataset[1].ntraces = dset.dataset[1].data.shape[0]


                    for freq_band in config["freq_bands"]:
                        # The clustering is performed separately in different frequency bands.
                        # The selections may change depending on the frequency band.
                        fmin, fmax = freq_band
                        twin_hw = config["hw_factor"] / fmin
                        # copy before filtering
                        dset.dataset[2] = CCData_serial(dset.dataset[1].data.copy(),
                                                        dset.dataset[1].timestamps.copy(),
                                                        dset.dataset[1].fs)
                        # whitening?
                        if config["do_whiten_cluster"]:
                            if config["whiten_nsmooth_cluster"] > 1:
                                fnorm = "rma"
                            else:
                                fnorm = "phase_only"
                            dset.post_whiten(f1=freq_band[0] * 0.75,
                                             f2=freq_band[1] * 1.5, stacklevel=2,
                                             npts_smooth=config["whiten_nsmooth_cluster"],
                                             freq_norm=fnorm)
                        # filter before clustering
                        dset.dataset[2].filter_data(filter_type=config["filt_type"],
                                         f_hp=fmin, f_lp=fmax, maxorder=config["filt_maxord"])
                        #window. The windows are all centered on lag 0 and extend to 10 / fmin
                        dset.dataset[2].window_data(t_mid=config["twin_mid"], hw=twin_hw,
                                         window_type="tukey", tukey_alpha=0.5,
                                         cutout=False)

                        # perform PCA on the random subset
                        dset.dataset[2].data = np.nan_to_num(dset.dataset[2].data)
                        X = StandardScaler().fit_transform(dset.dataset[2].data)
                        pca_rand = run_pca(X, min_cumul_var_perc=config["expl_var"])
                        # pca output is a scikit learn PCA object
                        # just for testing, run the Gaussian mixture here
                        # gm = gmm(pca_rand.transform(X), range(1, 12))

                        all_pccs = []
                        all_timestamps = []

                        # now go through all files again, read, filter, window, and fit the pcs
                        for datafile in datafiles:
                            print("Rank {} clustering {}-{} Hz band.".format(rank, *freq_band))
                            dset.add_datafile(datafile)
                            dset.data_to_memory(keep_duration=0)
                            # whitening?
                            if config["do_whiten_cluster"]:
                                if config["whiten_nsmooth_cluster"] > 1:
                                    fnorm = "rma"
                                else:
                                    fnorm = "phase_only"
                                dset.post_whiten(f1=freq_band[0] * 0.75,
                                                 f2=freq_band[1] * 1.5, stacklevel=0,
                                                 npts_smooth=config["whiten_nsmooth_cluster"],
                                                 freq_norm=fnorm)
                            # filter
                            dset.dataset[0].filter_data(filter_type=config["filt_type"],
                                             f_hp=fmin, f_lp=fmax, maxorder=config["filt_maxord"])
                            # window
                            dset.dataset[0].window_data(t_mid=config["twin_mid"], hw=twin_hw,
                                             window_type="tukey", tukey_alpha=0.5,
                                             cutout=False)
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
                        range_ncomps = range(1, config["nclustmax"] + 1)
                        gmmodels, n_clusters, gmixfinPCA, probs, BICF = gmm(all_pccs, range_ncomps)

                        # save the cluster labels
                        labels = np.zeros((2, len(all_timestamps)))
                        labels[0] = all_timestamps
                        labels[1] = gmixfinPCA
                        outputfile = "{}.{}-{}.{}_{}-{}Hz.gmmlabels.npy".format(station1, ch1, station2, ch2, fmin, fmax)
                        np.save(os.path.join(config["cluster_dir"], outputfile), labels)
    return()
