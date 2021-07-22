# -*- coding: utf-8 -*-
from ruido.utils.Function_Clustering_DFs import run_pca, gmm
import numpy as np
from ruido.classes.cc_dataset_serial import CCDataset_serial, CCData_serial
import os
from glob import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# read input

def run_clustering(config, rank, size, comm):
    if rank == 0:
        print("*"*80)
        print("Running clustering.")
        print("*"*80)
    else:
        pass

    # loop over stations
    to_do = []
    # parallelism: Distribute by input channel list
    for station1 in config["stations"]:
        for station2 in config["stations"]:
            # loop over components
            for ixch, ch1 in enumerate(config["channels"]):
                for ch2 in config["channels"][ixch:]:
                    if station1 != station2 and config["only_singlestation"]:
                        continue
                    # check if this is an autocorrelation and if yes, continue if drop_autocorrelations is true
                    if config["drop_autocorrelations"] and ch1 == ch2:
                        continue
                    to_do.append([station1, station2, ch1, ch2])
    print("Rank {}, to do length {}, from {} to {}.".format(rank, len(to_do), to_do[0], to_do[-1]))
    for id_to_do in to_do[rank::size]:
        station1 = id_to_do[0]
        station2 = id_to_do[1]
        ch1 = id_to_do[2]
        ch2 = id_to_do[3]
        # channel id
        channel_id = "{}.{}-{}.{}".format(station1, ch1, station2, ch2)
        if config["print_debug"]:
            print("Rank {} working on {}.".format(rank, channel_id))

        # find input files by glob
        datafiles = glob(os.path.join(config["input_directories"],
                                      "*.{}.*.{}--*.{}.*.{}.*windows.h5".format(station1,
                                                              ch1,
                                                              station2,
                                                              ch2)))
        if len(datafiles) == 0:
            print("Rank {}: No files found for {}.".format(rank, channel_id))
            continue
        datafiles.sort()

        if config["print_debug"]:
            print(datafiles)

        for ixfile, dfile in enumerate(datafiles):
            if ixfile == 0:
                # set up the dataset for the first time
                dset = CCDataset_serial(dfile)
                dset.data_to_memory()
            else:
                # read the data in
                dset.add_datafile(dfile)
                dset.data_to_memory(keep_duration=0)
            # use a random subset of each file (unless "all" requested)
            if type(config["n_samples_each_file"]) == int:
                ixs_random = np.random.choice(np.arange(dset.dataset[0].ntraces),
                                              min(config["n_samples_each_file"],
                                              dset.dataset[0].ntraces))
            elif config["n_samples_each_file"] == "all":
                ixs_random = np.arange(dset.dataset[0].ntraces)
            else:
                raise ValueError("n_samples_each_file must be an integer or \"all\".")


            if ixfile == 0:
                # create dataset on level 1
                dset.dataset[1] = CCData_serial(dset.dataset[0].data[ixs_random].copy(),
                                                dset.dataset[0].timestamps[ixs_random].copy(),
                                                dset.dataset[0].fs)
            else:
                # keep the randomly selected windows under key 1, 
                # adding to the previously selected ones
                dset.dataset[1].data = np.concatenate((dset.dataset[1].data,
                                                       dset.dataset[0].data[ixs_random].copy()))
                dset.dataset[1].timestamps = np.concatenate((dset.dataset[1].timestamps,
                                                             dset.dataset[0].timestamps[ixs_random].copy()))
                dset.dataset[1].ntraces = dset.dataset[1].data.shape[0]

        # now we have collected the randomly selected traces from all files to run the PCA on
        if config["print_debug"]:
            print(dset)

        # loop over frequency bands
        for freq_band in config["freq_bands"]:
            # The clustering is performed separately in different frequency bands.
            # The selections may change depending on the frequency band.
            fmin, fmax = freq_band
            twin_hw = config["hw_factor"] / fmin
            
            # determine output and if this has been computed already
            outputfile = "{}.{}-{}.{}_{}-{}Hz.gmmlabels.npy".format(station1, ch1, station2, ch2,
                                                                    fmin, fmax)
            if os.path.exists(os.path.join(config["cluster_dir"], outputfile)):
                if config["print_debug"]:
                    print("File {} has already been computed, continuing...".format(outputfile))
                continue


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
            # check if there are nans
            if config["print_debug"]:
                print("Are there nans? {}".format({1:"yes", 0:"no"}[np.any(np.isnan(dset.dataset[2].data))]))
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
                                            f_hp=fmin, f_lp=fmax,
                                            maxorder=config["filt_maxord"])
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
            labels = np.zeros((3, len(all_timestamps)))
            labels[0] = all_timestamps
            labels[1] = gmixfinPCA
            labels[2] = np.max(probs, axis=1)
            np.save(os.path.join(config["cluster_dir"], outputfile), labels)
    return()
