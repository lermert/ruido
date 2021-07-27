from ruido.classes.cc_dataset_mpi import CCDataset
try:
    from cmcrameri import cm
except:
    pass
import matplotlib.pyplot as plt ## for other colormaps
from glob import glob
import os
import re

# ----------------------------------------------------------------------------
# input
# ----------------------------------------------------------------------------
# input directory with *stacks_....h5 files
input_directory = "."#"output/stacks"
output_directory = "output/plots/stack_plots"
cmap = cm.broc
mask_gaps = True
step = 864000.
label_style = "month"
scale_factor_plotting = 0.1  # modifies the clipping of the color scale
figsize = (8.0, 4.5)  # controls the size of the plots
# ----------------------------------------------------------------------------
# end input
# ----------------------------------------------------------------------------

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# find files
files = glob(os.path.join(input_directory, "*stacks*.h5"))

for f in files:
    # read them
    dset = CCDataset(f)
    dset.data_to_memory()
    # figure out max. lag
    f0 = dset.datafile["stats"].attrs["f0_Hz"]
    maxlag_plot = 5./f0
    # plot them 
    outfile = os.path.join(output_directory, re.sub("h5", "png", os.path.basename(f)))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot("111")
    dset.plot_stacks(stacklevel=0, cmap=cmap, mask_gaps=mask_gaps,
                     step=step, normalize_all=True, label_style=label_style,
                     seconds_to_start=-maxlag_plot, seconds_to_show=maxlag_plot,
                     scale_factor_plotting=scale_factor_plotting, ax=ax)
    plt.tight_layout()
    plt.savefig(outfile)

    outfile1 = os.path.join(output_directory, re.sub("\.h5", "_poslag.png", os.path.basename(f)))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot("111")
    dset.plot_stacks(stacklevel=0, cmap=cmap, mask_gaps=mask_gaps,
                     step=step, normalize_all=True, label_style=label_style,
                     seconds_to_start=0.0, seconds_to_show=maxlag_plot,
                     scale_factor_plotting=scale_factor_plotting, ax=ax)
    plt.tight_layout()
    plt.savefig(outfile1)


    outfile2 = re.sub("poslag", "neglag", outfile1)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot("111")
    dset.plot_stacks(stacklevel=0, cmap=cmap, mask_gaps=mask_gaps,
                     step=step, normalize_all=True, label_style=label_style,
                     seconds_to_start=-maxlag_plot, seconds_to_show=0.0,
                     scale_factor_plotting=scale_factor_plotting, ax=ax)
    plt.tight_layout()
    plt.savefig(outfile2)

    plt.close()

    
