import numpy as np
import seaborn as sns
from obspy import UTCDateTime
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
import re

# ----------------------------------------------------------------------------
# input
# ----------------------------------------------------------------------------
# input directory with *gmmlabels.npy files
input_directory = "/home/lermert/Desktop/CDMX/clustering/results_from_uwork/"
#"output/clusters/"
output_directory = "output/plots/clusterlabel_plots"
n_max = 100000  # maybe don't plot all the labels, that will be slow and need lots of memory
# save output plots (if False, they will be shown interactively)
save_plots = False
# ----------------------------------------------------------------------------
# end input
# ----------------------------------------------------------------------------
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

cluster_files = glob(os.path.join(input_directory, "*.gmmlabels.npy"))

for cfile in cluster_files:
    clusters = np.load(cfile)[:, 0:n_max]

    dat = pd.DataFrame(columns=["timestamps", "labels"])
    dat["timestamps"] = clusters[0]
    dat["labels"] = clusters[1]
    dat["hour"] = [float(UTCDateTime(t).strftime("%H")) for t in dat["timestamps"]]
    sns.scatterplot(x="timestamps", y="hour", data=dat, hue="labels", linewidth=0,
                    palette=sns.color_palette(n_colors=len(dat.labels.unique())),
                    )


    years = []
    xticklabels = []
    xticks = []
    for tst in dat["timestamps"].unique():
        if UTCDateTime(tst).strftime("%Y") in years:
            continue
        years.append(UTCDateTime(tst).strftime("%Y"))
        xticklabels.append(UTCDateTime(tst).strftime("%Y/%m"))
        xticks.append(tst)
    plt.xlabel("Date")
    plt.xticks(xticks, xticklabels, rotation=45)

    if save_plots:
        outfile = os.path.join(output_directory, re.sub("npy", "png", os.path.basename(cfile)))
        plt.savefig(outfile)
        plt.close()
    else:
        plt.show()

