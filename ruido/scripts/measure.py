from ruido.classes.cc_dataset_mpi import CCDataset, CCData
from ruido.scripts.measurements import run_measurement
from ruido.utils.read_config import read_config
from obspy import UTCDateTime
import os
import numpy as np
import pandas as pd
import time
from glob import glob
import re
import yaml
import io
import sys
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

configfile = sys.argv[1]
config = read_config(configfile)

input_files = glob(os.path.join(config["stack_dir"], "*stacks_*.h5"))
input_files.sort()
if input_files == []:
    raise ValueError("No input files found")


# For each input file:
# Read in the stacks
# For each time window:
# measurement
# save the result
# set up the dataframe to collect the results
output = pd.DataFrame(columns=["timestamps", "t0_s", "t1_s", "f0_Hz",  "f1_Hz",
                               "tag", "dvv_max", "dvv", "cc_before", "cc_after",
                               "dvv_err", "cluster"])

for iinf, input_file in enumerate(input_files):
    ixf = int(os.path.splitext(input_file)[0].split("_")[-2][2:])
    cl_label = int(os.path.splitext(input_file)[0].split("_")[-1][2:])
    station = os.path.basename(input_file.split(".")[1])
    ch1 = os.path.basename(input_file.split(".")[2][0: 3])
    ch2 = os.path.basename(input_file.split(".")[4])
    freq_band = config["freq_bands"][ixf]

    # read into memory
    dset = CCDataset(input_file)
    dset.data_to_memory()

    # interpolate and plot the stacks
    if rank == 0:
        if dset.dataset[0].fs != config["new_fs"]:
            dset.dataset[0].interpolate_stacks(new_fs=config["new_fs"])

        if config["do_plots"]:
            plot_output = re.sub("\.h5", "_{}.png".format(ixf),
                                 os.path.basename(input_file))
            dset.plot_stacks(stacklevel=0, label_style="year",
                             seconds_to_show=config["plot_tmax"][ixf],
                             outfile=plot_output)

    # find max. dvv that will just be short of a cycle skip
    # then extend by skipfactor
    for twin in config["twins"][ixf]:
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

        print(dset)

        output_table = run_measurement(dset, config, twin, freq_band, rank, comm)
        if rank == 0:
            output_table["cluster"] = np.ones(len(output_table)) * cl_label
        output = pd.concat([output, output_table], ignore_index=True)

        comm.barrier()
        if rank == 0:
            del dset.dataset[1]
            del dset.dataset[2]
        else:
            pass

# at the end write all to file
if rank == 0:
    outfile_name = "{}_{}{}_{}_{}.csv".format(station, ch1, ch2, config["measurement_type"],
                                              config["reference_type"])
    output.to_csv(os.path.join(config["msr_dir"], outfile_name))
