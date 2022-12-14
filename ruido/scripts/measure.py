from ruido.classes.cc_dataset_mpi import CCDataset, CCData
from ruido.scripts.measurements import run_measurement
import os
import numpy as np
import pandas as pd
from glob import glob


def run_measure(config, rank, size, comm):

    if rank == 0:
        print("*"*80)
        print("Running measurement.")
        print("*"*80)

    if config["print_debug"]:
        print("Rank {} is working on measurement.".format(rank))

    corrtype = config["correlation_type"]
    # loop over stations and channels, to produce 1 output file per channel
    for sta in config["stations"]:

        for ch1 in config["channels"]:
            for ch2 in config["channels"]:


                input_files = glob(os.path.join(config["stack_dir"], "*.{}.*.{}--*.{}.*.{}.{}.stacks_*.h5".format(sta, ch1, sta, ch2, corrtype)))
                print(os.path.join(config["stack_dir"], "*.{}.*.{}--*.{}.*.{}.{}.stacks_*.h5".format(sta, ch1, sta, ch2, corrtype)))
                input_files.sort()

                if config["print_debug"] and rank == 0:
                    print(input_files)

                if len(input_files) == 0:
                    print("No input files found for: ", sta, ch1, ch2)
                    continue


                # start a new output table.
                output = pd.DataFrame(columns=["timestamps", "t0_s", "t1_s", "f0_Hz",  "f1_Hz",
                                               "tag", "dvv_max", "dvv", "cc_before", "cc_after",
                                               "dvv_err", "cluster"])
                
                
                # For each input file:
                # Read in the stacks
                    # For each time window:
                    # measurement
                    # save the result

                for iinf, input_file in enumerate(input_files):
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

                    station1 = os.path.basename(input_file.split(".")[1])
                    station2 = os.path.basename(input_file.split(".")[4])

                    ch_id = "{}.{}-{}.{}".format(station1, ch1, station2, ch2)

                    # read into memory
                    dset = CCDataset(input_file)
                    dset.data_to_memory()

                    # interpolate and plot the stacks
                    if rank == 0:
                        if dset.dataset[0].fs != config["new_fs"]:
                            dset.dataset[0].interpolate_stacks(new_fs=config["new_fs"])

                    # find max. dvv that will just be short of a cycle skip
                    # then extend by skipfactor
                    ixf = config["freq_bands"].index(freq_band)
                    for twin in config["twins"][ixf]:

                        # get the time window for this station pair
                        if rank == 0:
                            print("Measurement window {}, {} s...".format(*twin))
                        else:
                            pass
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
                            dset.dataset[1].window_data(t_mid=t_mid, hw=hw, window_type=config["window_type"])
                        else:
                            pass

                        output_table = run_measurement(dset, config, twin, freq_band, rank, comm)
                        if rank == 0 and config["use_clusters"]:
                            output_table["cluster"] = np.ones(len(output_table)) * cl_label
                        else:
                            pass
                        output = pd.concat([output, output_table], ignore_index=True)

                        comm.Barrier()
                        if rank == 0:
                            del dset.dataset[1]
                            try:
                                del dset.dataset[2]
                            except KeyError:
                                pass
                        else:
                            pass
                # at the end write all to file (the last channel pair)
                if rank == 0:
                    outfile_name = "{}_{}_{}.csv".format(ch_id, config["measurement_type"],
                                                         config["reference_type"])
                    output.to_csv(os.path.join(config["msr_dir"], outfile_name))
