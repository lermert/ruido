import yaml
from obspy import UTCDateTime
import os

def read_config(configfile):
    # get configuration
    config = yaml.safe_load(open(configfile))

    # CHECKS
    assert (type(config["ngrid"]) == int), "ngrid must be integer in config.yml"
    
    for fp in config["freq_bands"]:
        assert len(fp) == 2, "Frequency bands must be specified as \
tuples of fmin, fmax"
    
    # ADDITIONAL DERIVED PARAMETERS
    if config["reference_type"] == "inversion":
        config["badwins"] = [[UTCDateTime(bt) for bt in entr] \
                              for entr in config["skiptimes_inversion"]]
    elif config["reference_type"] == "bootstrap":
        config["r_duration"] = config["reference_length_days"] * 86400.
    elif config["reference_type"] == "list":
        config["r_windows"] = [[UTCDateTime(reft) for reft in entr]\
                                for entr in config["reference_list"]]
    elif config["reference_type"] == "trailing":
        pass
    else:
        raise ValueError("Unknown reference type {}".format(config["reference_type"]))

    config["t0"] = UTCDateTime(config["t0"])
    config["t1"] = UTCDateTime(config["t1"])
    

    # define the time windows
    config["plot_tmax"] = []
    config["twins"] = []
    for freq_band in config["freq_bands"]:
        longest_t = 1. / freq_band[0]
        twinsf = []
        twinsf.append([-20. * longest_t, -8. * longest_t])
        twinsf.append([-10. * longest_t, -4. * longest_t])
        twinsf.append([ 4. * longest_t, 10. * longest_t])
        twinsf.append([ 8. * longest_t, 20. * longest_t])
        config["twins"].append(twinsf)
        config["plot_tmax"].append(10.0 * longest_t)


    # directory setup
    for dtry in [config["stack_dir"], config["msr_dir"],
                 config["cluster_dir"]]:
        try:
            os.makedirs(dtry)
        except FileExistsError:
            pass
        # copy metadata
        t_now = UTCDateTime().strftime("%Y-%m-%dT%H.%M")
        with open(os.path.join(dtry, "configfile_{}.yml".format(t_now)), "w") as coutf:
            coutf.write(yaml.dump(config))


    return(config)