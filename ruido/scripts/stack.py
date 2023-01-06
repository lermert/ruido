# coding: utf-8
from ruido.classes.cc_dataset_mpi import CCDataset
from obspy import UTCDateTime
import time
import numpy as np
import os
import h5py
from glob import glob
from warnings import warn

def add_stacks(dset, config, rank):

    if rank != 0:
        raise ValueError("serial function")
    # make a difference whether there are cluster labels or not.
    # if there are then use them for selection.
    if len(dset.dataset) == 0:
        print("Nothing to stack. Call data_to_memory first")
        return()

    if dset.dataset[0].cluster_labels is not None:

        for clabel in np.unique(dset.dataset[0].cluster_labels):
            if clabel == -1:  # de-selected windows
                continue
            try:
                t_running = dset.dataset[clabel + 1].timestamps.max() + config["step"]
            except KeyError:
                t_running = max(config["t0"], dset.dataset[0].timestamps.min())

            # if t_running_in is None:
            #     t_running = max(config["t0"], dset.dataset[0].timestamps.min())
            # else:
            #     t_running = t_running_in

            while t_running < min(config["t1"], dset.dataset[0].timestamps.max()):

                stimes = dset.dataset[0].group_for_stacking(t_running, duration=config["duration"],
                                                            cluster_label=clabel)
                stimes = dset.dataset[0].select_for_stacking(stimes, "rms_percentile",
                                                             perc=config["percentile_rms"],
                                                             mode="upper")
                if stimes == []:
                    print("No windows, ", UTCDateTime(t_running))
                    t_running += config["step"]
                    continue
                if len(stimes) < config["minimum_stack_len"]:
                    print("Not enough windows, ", UTCDateTime(t_running))
                    t_running += config["step"]
                    continue

                dset.stack(np.array(stimes), stackmode=config["stackmode"],
                           epsilon_robuststack=config["robuststack_epsilon"],
                           stacklevel_out=clabel+1)
                t_running += config["step"]

    else:
        # if t_running_in is None:
        #     t_running = max(config["t0"], dset.dataset[0].timestamps.min())
        # else:
        #     t_running = t_running_in
        if len(dset.dataset) > 1:
            t_running = dset.dataset[1].timestamps.max() + config["step"]
        else:
            t_running = max(config["t0"], dset.dataset[0].timestamps.min())

        while t_running < min(config["t1"], dset.dataset[0].timestamps.max()):
            stimes = dset.dataset[0].group_for_stacking(t_running, duration=config["duration"])
            stimes = dset.dataset[0].select_for_stacking(stimes, "rms_percentile",
                                                         perc=config["percentile_rms"], mode="upper")
            if stimes == []:
                t_running += config["step"]
                continue
            if len(stimes) < config["minimum_stack_len"]:
                t_running += config["step"]
                continue

            dset.stack(np.array(stimes), stackmode=config["stackmode"],
                       epsilon_robuststack=config["robuststack_epsilon"])
            t_running += config["step"]


def run_stacking(config, rank, size, comm):
    if rank == 0:
        print("*"*80)
        print("Running stacking.")
        print("*"*80)

    # loop over frequency bands
    for ixf, freq_band in enumerate(config["freq_bands"]):
        ids_done = []
        # loop over components:
        for station1 in config["stations"]:
            for station2 in config["stations"]:
                for ixch1, ch1 in enumerate(config["channels"]):
                    for ch2 in config["channels"]:
                        channel_id = "{}.{}-{}.{}".format(station1, ch1, station2, ch2)
                        if channel_id in ids_done:
                            continue
                        else:
                            ids_done.append(channel_id)
                            if config["print_debug"] and rank == 0:
                                print(channel_id)

                        if ch1 == ch2 and config["drop_autocorrelations"]:
                            continue

                        input_files = glob(os.path.join(config["input_directories"],
                                                        "*.{}.*.{}--*.{}.*.{}.*{}.windows.h5".format(station1,
                                                                                          ch1,
                                                                                          station2,
                                                                                          ch2, config["correlation_type"])))
                        if len(input_files) == 0:
                            if config["print_debug"] and rank == 0:
                                print("No input for this channel.")
                            continue
                        # VERY IMPORTANT: sort so that e.g.
                        # chronologic order of input files is preserved
                        input_files.sort()
                        if config["print_debug"] and rank == 0:
                            print(input_files)


                        if config["use_clusters"]:
                            clusterfile = os.path.join(config["cluster_dir"],
                                                       "{}.{}-{}.{}_{}-{}Hz.gmmlabels.npy".format(
                                                        station1, ch1, station2, ch2, freq_band[0],
                                                        freq_band[1]))
                            try:
                                clusters = np.load(clusterfile)
                            except FileNotFoundError:
                                continue
                        # read in the data, one file at a time, adding stacks as we go along
                        dset = CCDataset(input_files[0])

                        for i, f in enumerate(input_files):
                            ctype = os.path.basename(f).split(".")[7]
                            if i == 0:
                                network = os.path.basename(f).split(".")[0]
                                dset.data_to_memory()
                                if rank == 0 and config["use_clusters"]:
                                    dset.dataset[0].add_cluster_labels(clusters)
                                else:
                                    pass
                                print("here")
                            else:
                                dset.add_datafile(f)
                                dset.data_to_memory(keep_duration=config["duration"])
                                if rank == 0 and config["use_clusters"]:
                                    dset.dataset[0].add_cluster_labels(clusters)
                                else:
                                    pass

                            if config["do_whiten"]:
                                if config["whiten_nsmooth"] > 1:
                                    fnorm = "rma"
                                else:
                                    fnorm = "phase_only"
                                dset.post_whiten(f1=freq_band[0] * 0.75,
                                                 f2=freq_band[1] * 1.5, stacklevel=0,
                                                 npts_smooth=config["whiten_nsmooth"],
                                                 freq_norm=fnorm)
                            dset.filter_data(f_hp=freq_band[0], f_lp=freq_band[1], taper_perc=0.2,
                                             filter_type=config["filt_type"], stacklevel=0,
                                             maxorder=config["filt_maxord"])
                            print("Filtering done")
                            if rank == 0:
                                # try:
                                #     t_running = dset.dataset[1].timestamps.max() + config["step"]
                                # except KeyError:
                                #     t_running = max(dset.dataset[0].timestamps.min(), config["t0"])
                                add_stacks(dset, config, rank)
                                print(dset)
                            else:
                                pass

                            comm.Barrier()

                        # save the stacks
                        if rank == 0:
                            for stacklevel in dset.dataset.keys():
                                if stacklevel == 0:
                                    continue
                                if dset.dataset[stacklevel].ntraces == 0:
                                    continue

                                if config["use_clusters"]:
                                    cltag = "_cl{}".format(stacklevel)
                                else:
                                    cltag = "_noclust"
                                outfile = "{}.{}..{}--{}.{}..{}.{}.stacks_{}days_{}-{}_{}-{}Hz{}.h5".format(
                                    network, station1, ch1, network, station2, ch2, ctype, config["duration"]//86400,
                                    UTCDateTime(config["t0"]).strftime("%Y"),
                                    UTCDateTime(config["t1"]).strftime("%Y"),
                                    freq_band[0], freq_band[1], cltag)
                                outfile = os.path.join(config["stack_dir"], outfile)
                                outfile = h5py.File(outfile, "w")
                                cwin = outfile.create_group("corr_windows")
                                stats = outfile.create_dataset("stats", data=())
                                stats.attrs["sampling_rate"] = dset.dataset[stacklevel].fs
                                stats.attrs["f0_Hz"] = freq_band[0]
                                stats.attrs["f1_Hz"] = freq_band[1]
                                stats.attrs["channel_id"] = channel_id
                                for k, v in config.items():
                                    # this will save the configuration in the output file.
                                    # it also saves the configuration for measurements which is not really relevant
                                    # but whatever
                                    if k in ["freq_bands"]:
                                        # no need to record -- have recorded already
                                        continue
                                    elif k in ["r_windows", "skiptimes_inversion", "freq_bands"]:
                                        ll = [[vvv for vvv in vv] for vv in v]
                                        outstr = len(ll)*"{},"
                                        stats.attrs[k] = outstr.format(*ll)
                                        print(outstr.format(*ll))
                                    elif k in ["t0", "t1"]:
                                        stats.attrs[k] = UTCDateTime(v).strftime("%Y.%j.%H.%M.%S")
                                    else:
                                        try:
                                            stats.attrs[k] = v
                                        except TypeError:
                                            print("could not write {}: {} to output file".format(k, v))
                                            pass

                                cwin.create_dataset("data", data=dset.dataset[stacklevel].data)
                                cwin.create_dataset("timestamps", data=dset.dataset[stacklevel].timestamps.flatten())
                                outfile.flush()
                                outfile.close()
