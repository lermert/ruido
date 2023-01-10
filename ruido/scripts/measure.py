from ruido.classes.cc_dataset_mpi import CCDataset, CCData
from ruido.scripts.measurements import run_measurement
import os
import numpy as np
import pandas as pd
from glob import glob
from obspy.geodetics import gps2dist_azimuth

def run_measure(config, rank, size, comm):

    if rank == 0:
        print("*"*80)
        print("Running measurement.")
        print("*"*80)

    if config["print_debug"]:
        print("Rank {} is working on measurement.".format(rank))

    stationlist = pd.read_csv(config["stationlist_file"])
    if rank == 0:
        print(stationlist)
    else:
        pass
    corrtype = config["correlation_type"]
    # loop over stations and channels, to produce 1 output file per channel
    for sta1 in config["stations"]:
        for sta2 in config["stations"]:
            for ch1 in config["channels"]:
                for ch2 in config["channels"]:


                    input_files = glob(os.path.join(config["stack_dir"], "*.{}.*.{}--*.{}.*.{}.{}.stacks_*.h5".format(sta1, ch1, sta2, ch2, corrtype)))
                    input_files.sort()

                    if config["print_debug"] and rank == 0:
                        print(input_files)

                    if len(input_files) == 0:
                        print("No input files found for: ", sta1, ch1, sta2, ch2)
                        continue
                    
                    # get the station distance
                    lat1 = stationlist[stationlist.Station==sta1].Latitude.values[0]
                    lon1 = stationlist[stationlist.Station==sta1].Longitude.values[0]
                    lat2 = stationlist[stationlist.Station==sta2].Latitude.values[0]
                    lon2 = stationlist[stationlist.Station==sta2].Longitude.values[0]
                    dist = gps2dist_azimuth(lat1, lon1, lat2, lon2)[0]; print(dist)

                    # start a new output table.
                    output = pd.DataFrame(columns=["timestamps", "t0_s", "t1_s", "f0_Hz",  "f1_Hz",
                                                "tag", "dvv_max", "dvv", "cc_before", "cc_after",
                                                "dvv_err", "cluster"])
                    
                    
                    # For each input file:
                    
                    # Read in the stacks
                        # For each time window:
                        # measurement
                        # save the result

                    for input_file in input_files:
                        f0 = float(input_file.split("_")[-2].split("-")[0])
                        f1 = float(input_file.split("_")[-2].split("-")[-1][:-2])
                        freq_band = [f0, f1]
                        
                        # check if the frequency band of the file is among the ones that should be measured
                        if freq_band not in config["freq_bands"]: continue

                        
                        if config["use_clusters"]:
                            try:
                                cl_label = int(os.path.splitext(input_file)[0].split("_")[-1][2:])
                            except ValueError:
                                assert type(os.path.splitext(input_file)[0].split("_")[-1][2:]) == str
                                continue

                        ch_id = "{}.{}-{}.{}".format(sta1, ch1, sta2, ch2)

                        # read into memory
                        dset = CCDataset(input_file)
                        dset.data_to_memory()

                        # interpolate and plot the stacks
                        if rank == 0:
                            if dset.dataset[0].fs != config["new_fs"]:
                                dset.dataset[0].interpolate_stacks(new_fs=config["new_fs"])
                        else:
                            pass 
                        # define the time windows:
                        # time offset
                        offset_t = dist / config["wave_velocity_mps"]
                        # here, we fix the offset to the nearest sample. If not, it's a mess to window the data
                        print(offset_t)
                        offset_t = dset.dataset[0].lag[np.argmin(np.abs(dset.dataset[0].lag - offset_t))]
                        print(offset_t)

                        # time window half-width
                        longest_T = 1. / min(freq_band)
                        # shift window to coda
                        offset_ts = [offset_t + wd * longest_T for wd in config["window_delays_in_multiples_of_longest_period"]]
                        win_hw = [hwmlt * longest_T for hwmlt in config["window_half_widths_in_multiples_of_longest_period"]]
                        # time windows
                        twins = []
                        for offset_t in offset_ts:
                            twins.extend([[offset_t, offset_t + 2*whw] for whw in win_hw])

                        if sta1 != sta2 or ch1 != ch2:
                            twins.extend([[-twin[1], -twin[0]] for twin in twins])
                        print(twins)

                        for twin in twins:

                            # get the time window for this station pair
                            if rank == 0:
                                print("Measurement window {}, {} s...".format(*twin))
                            else:
                                pass
                            # find max. dvv that will just be short of a cycle skip
                            # then extend by "skipfactor"
                            maxdvv = config["skipfactor"] * 1. / (2. * freq_band[1] *
                                                                max(abs(np.array(twin))))
                            config["maxdvv"] = maxdvv

                            # window
                            t_mid = (twin[0] + twin[1]) / 2.
                            hw = (twin[1] - twin[0]) / 2.

                            if rank == 0:
                                dset.dataset[1] = CCData(dset.dataset[0].data.copy(),
                                                        dset.dataset[0].timestamps.copy(),
                                                        dset.dataset[0].fs)
                                dset.dataset[1].window_data(t_mid=t_mid, hw=hw, window_type=config["window_type"], cutout=True)
                                lwin = [dset.dataset[1].lag[0], dset.dataset[1].lag[-1]]
                            else:
                                lwin = []

                            lwin = comm.bcast(lwin, root=0)

                            output_table = run_measurement(dset, config, lwin, freq_band, rank, comm)
                            if rank == 0 and config["use_clusters"]:
                                output_table["cluster"] = np.ones(len(output_table)) * cl_label
                            else:
                                pass

                            if rank == 0:
                                output = pd.concat([output, output_table], ignore_index=True)
                            else:
                                pass
                            

                            comm.Barrier()

                    # write to file
                    if rank == 0:
                        outfile_name = "{}_{}_{}.csv".format(ch_id, config["measurement_type"],
                                                            config["reference_type"])
                        output.to_csv(os.path.join(config["msr_dir"], outfile_name))
                        print("Done with {}".format(ch_id))
                    else:
                        pass