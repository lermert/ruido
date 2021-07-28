import numpy as np
import pandas as pd
from obspy import UTCDateTime
from ruido.classes.cc_dataset_mpi import CCData

def measurement_brenguier(dset, conf, twin, freq_band, rank, comm):
    if rank == 0:
        tstmps = dset.dataset[1].timestamps
        # cut out times where the station wasn't operating well
        # define in measurement_config.yml
        bad_ixs = []
        for badwindow in conf["badwins"]:
            ixbw1 = np.argmin((tstmps - badwindow[0]) ** 2)
            ixbw2 = np.argmin((tstmps - badwindow[1]) ** 2)
            bad_ixs.extend(list(np.arange(ixbw1, ixbw2)))

        good_windows = [ixwin for ixwin in range(len(tstmps)) if not ixwin in bad_ixs]
        data = dset.dataset[1].data[good_windows]
        tstmps = dset.dataset[1].timestamps[good_windows]
        dset.dataset[2] = CCData(data, tstmps, dset.dataset[1].fs)
        n = len(dset.dataset[2].timestamps)
        k = int(n * (n - 1) / 2.)
        data_dvv = np.zeros(k)
        data_dvv_err = np.zeros(k)
    else:
        n = 0

    n = comm.bcast(n, root=0)
    counter = 0
    for i in range(n):
        # i-th stack as reference
        if rank == 0:
            ref = dset.dataset[2].data[i, :]
        else:
            ref = None
        ref = comm.bcast(ref, root=0)

        dvv, dvv_timest, ccoeff, \
            best_ccoeff, dvv_error = dset.measure_dvv_par(f0=freq_band[0], f1=freq_band[1], ref=ref, stacklevel=2,
                                                          ngrid=100, method=conf["measurement_type"],
                                                          dvv_bound=maxdvv)
        comm.barrier()
        if rank == 0:
            for j in range(i + 1, n):
                data_dvv[counter] = dvv[j]
                data_dvv_err[counter] = dvv_error[j]
                counter += 1
        else:
            pass
    return(dvv_timest, dvv, ccoeff, best_ccoeff, dvv_error, None)

def measurement_incremental(dset, config, twin, freq_band, rank, comm,
                            stacklevel=1):
 
    if rank == 0:
        references = np.cumsum(dset.dataset[stacklevel].data, axis=0)
        a = np.arange(1, references.shape[0] + 1)
        b = np.ones(references.shape)
        c = (a * b.T).T
        references /= c
        stacks = dset.dataset[stacklevel].data
        timestamps = dset.dataset[stacklevel].timestamps
        fs = dset.dataset[stacklevel].fs
        ntraces = dset.dataset[stacklevel].ntraces

    else:
        references = None
        stacks = None
        timestamps = None
        fs = 0
        ntraces = 0
    references = comm.bcast(references, root=0)
    timestamps = comm.bcast(timestamps, root=0)
    stacks = comm.bcast(stacks, root=0)
    fs = comm.bcast(fs, root=0)
    ntraces = comm.bcast(ntraces, root=0)

    size = comm.Get_size()
    t = np.zeros(ntraces)
    dvv = np.zeros(ntraces)
    cc0 = np.zeros(ntraces)
    cc1 = np.zeros(ntraces)
    err = np.zeros(ntraces)
    tags = np.zeros(ntraces)

    if rank > 0:
        dset.dataset[stacklevel] = CCData(stacks, timestamps, fs)
    else:
        pass

    for ix in range(rank, ntraces, size):
        if ix > 0:
            ref = references[ix-1, :]
        else:
            ref = references[0, :]
        dvvp, dvv_timestp, ccoeffp, \
            best_ccoeffp, dvv_errorp, \
                   = dset.measure_dvv_ser(f0=freq_band[0], f1=freq_band[1], stacklevel=stacklevel,
                                          ref=ref, ngrid=config["ngrid"],
                                           method=config["measurement_type"], indices=[ix],
                                           dvv_bound=config["maxdvv"],
                                             )
        t[ix] = dvv_timestp[0]
        dvv[ix] = dvvp[0]
        cc0[ix] = ccoeffp[0]
        cc1[ix] = best_ccoeffp[0]
        err[ix] = dvv_errorp[0]

    comm.barrier()

    # now collect
    if rank == 0:
        t_all = t.copy()
        dvv_all = dvv.copy()
        cc0_all = cc0.copy()
        cc1_all = cc1.copy()
        err_all = err.copy()
        for other_rank in range(1, size):
            comm.Recv(t, source=other_rank, tag=77)
            t_all += t
            comm.Recv(dvv, source=other_rank, tag=78)
            dvv_all += dvv
            comm.Recv(cc0, source=other_rank, tag=79)
            cc0_all += cc0
            comm.Recv(cc1, source=other_rank, tag=80)
            cc1_all += cc1
            comm.Recv(err, source=other_rank, tag=81)
            err_all += err
    else:
        comm.Send(t, dest=0, tag=77)
        comm.Send(dvv, dest=0, tag=78)
        comm.Send(cc0, dest=0, tag=79)
        comm.Send(cc1, dest=0, tag=80)
        comm.Send(err, dest=0, tag=81)


    if rank == 0:
        return(timestamps, dvv_all, cc0_all, cc1_all, err_all, tags)
    else:
        return([], [], [], [], [], [])


def measurement_trailing(dset, config, twin, freq_band, rank, comm,
                         stacklevel=1, restack_duration=30. * 86400):
    if rank == 0:
        ntraces = dset.dataset[stacklevel].ntraces
        nt = dset.dataset[stacklevel].npts
        stacks = dset.dataset[stacklevel].data
        timestamps = dset.dataset[stacklevel].timestamps
        fs = dset.dataset[stacklevel].fs
        references = np.zeros(dset.dataset[stacklevel].data.shape)
        dset.dataset[2] = CCData(dset.dataset[1].data[0: 1].copy(),
                                 timestamps[0], fs)
        for i_d in range(1, ntraces):
            ixs_stack = dset.dataset[0].group_for_stacking(t0=dset.dataset[stacklevel].timestamps[i_d] - restack_duration,
                                                           duration=restack_duration)
            a = 2
            while len(ixs_stack) == 0:
                ixs_stack = dset.dataset[0].group_for_stacking(t0=dset.dataset[stacklevel].timestamps[i_d] - restack_duration,
                                                               duration=restack_duration * a)
                a *= 2.
            dset.stack(ixs_stack, stacklevel_in=1, stacklevel_out=2)
            ixs_stack_old = ixs_stack
        references = dset.dataset[2].data
        
    else:
        references = None
        stacks = None
        timestamps = None
        fs = 0
        ntraces = 0
    references = comm.bcast(references, root=0)
    timestamps = comm.bcast(timestamps, root=0)
    stacks = comm.bcast(stacks, root=0)
    fs = comm.bcast(fs, root=0)
    ntraces = comm.bcast(ntraces, root=0)

    size = comm.Get_size()
    t = np.zeros(ntraces)
    dvv = np.zeros(ntraces)
    cc0 = np.zeros(ntraces)
    cc1 = np.zeros(ntraces)
    err = np.zeros(ntraces)
    tags = np.zeros(ntraces)

    if rank > 0:
        dset.dataset[stacklevel] = CCData(stacks, timestamps, fs)
    else:
        pass

    for ix in range(rank, ntraces, size):
        if ix > 0:
            ref = references[ix-1, :]
        else:
            ref = references[0, :]
        dvvp, dvv_timestp, ccoeffp, \
            best_ccoeffp, dvv_errorp, \
            = dset.measure_dvv_ser(f0=freq_band[0], f1=freq_band[1],
                                   stacklevel=stacklevel,
                                   ref=ref, ngrid=config["ngrid"],
                                   method=config["measurement_type"],
                                   indices=[ix],
                                   dvv_bound=config["maxdvv"],
                                   )
        t[ix] = dvv_timestp[0]
        dvv[ix] = dvvp[0]
        cc0[ix] = ccoeffp[0]
        cc1[ix] = best_ccoeffp[0]
        err[ix] = dvv_errorp[0]

    comm.barrier()

    # now collect
    if rank == 0:
        t_all = t.copy()
        dvv_all = dvv.copy()
        cc0_all = cc0.copy()
        cc1_all = cc1.copy()
        err_all = err.copy()
        for other_rank in range(1, size):
            comm.Recv(t, source=other_rank, tag=77)
            t_all += t
            comm.Recv(dvv, source=other_rank, tag=78)
            dvv_all += dvv
            comm.Recv(cc0, source=other_rank, tag=79)
            cc0_all += cc0
            comm.Recv(cc1, source=other_rank, tag=80)
            cc1_all += cc1
            comm.Recv(err, source=other_rank, tag=81)
            err_all += err
    else:
        comm.Send(t, dest=0, tag=77)
        comm.Send(dvv, dest=0, tag=78)
        comm.Send(cc0, dest=0, tag=79)
        comm.Send(cc1, dest=0, tag=80)
        comm.Send(err, dest=0, tag=81)

    if rank == 0:
        return(timestamps, dvv_all, cc0_all, cc1_all, err_all, tags)
    else:
        return([], [], [], [], [], [])


def measurement_list(dset, config, twin, freq_band, rank, comm):
    return_empty = False

    if config["measurement_type"] == "cc_timeshift":
        assert config["reference_type"] == "list", "Run CC timeshift only with a list of references"
        
    if rank == 0:
        data = dset.dataset[1]
        if config["reference_type"] == "list":
            ntraces = data.ntraces
            nt = data.npts
            stacks = data.data
            timestamps = data.timestamps
            fs = data.fs
            for r_window in config["r_windows"]:

                ixs_stack = dset.dataset[0].group_for_stacking(t0=r_window[0].timestamp,
                                                               duration=r_window[1].timestamp - r_window[0].timestamp)
                dset.stack(ixs_stack, stacklevel_in=1, stacklevel_out=2)
            try:
                references = dset.dataset[2].data
            except KeyError:
                references = None
                return_empty = True
        elif config["reference_type"] == "bootstrap":
            ntraces = data.ntraces
            nt = data.npts
            stacks = data.data
            timestamps = data.timestamps
            fs = data.fs
            bootstrap_n = config["bootstrap_samples"]
            ref_duration = config["r_duration"]
            references = []
            tstmps_bs = dset.dataset[1].timestamps

            if config["bootstrap_type"] == "consecutive":
                last_ix = np.argmin(np.abs(tstmps_bs - tstmps_bs[-1] + ref_duration))
                if len(tstmps_bs[:last_ix]) > 1:
                    tstmps_bs = tstmps_bs[:last_ix]
                for i in range(bootstrap_n):
                    if len(tstmps_bs) > 1:
                        tref = np.random.choice(tstmps_bs)
                    else:
                        print("Useless case: 1 or less windows found in chosen reference.")
                        print("{}-{}s, {}-{} Hz, {}".format(*twin, *freq_band, UTCDateTime(tstmps_bs[0])))
                        return_empty = True
                        tref = tstmps_bs[0]
                    
                    if config["print_debug"]:
                        print("random t: ", UTCDateTime(tref))
                    ws_ref = dset.dataset[0].group_for_stacking(t0=tref,
                                                     duration=ref_duration,
                                                     )
                    dset.stack(ws_ref, stacklevel_in=1, stacklevel_out=2)
                    references.append(dset.dataset[2].data[-1, :].copy())
            else:
                for i in range(bootstrap_n):
                    bsixs = np.random.choice(np.arange(len(tstmps_bs)),
                            config["bootstrap_n_randomwindows"])
                    dset.stack(bsixs, stacklevel_in=1, stacklevel_out=2)
                    references.append(dset.dataset[2].data[-1, :].copy())
            references = np.array(references)

    else:
        references = None
        stacks = None
        timestamps = None
        fs = 0
        ntraces = 0

    return_empty = comm.bcast(return_empty, root=0)
    if return_empty:
        return([], [], [], [], [], [])
    references = comm.bcast(references, root=0)
    timestamps = comm.bcast(timestamps, root=0)
    stacks = comm.bcast(stacks, root=0)
    fs = comm.bcast(fs, root=0)
    ntraces = comm.bcast(ntraces, root=0)

    t_list = []
    dvv_list = []
    cc0_list = []
    cc1_list = []
    err_list = []
    tags_list = []

    for ix_ref, ref in enumerate(references):

        size = comm.Get_size()
        t = np.zeros(ntraces)
        dvv = np.zeros(ntraces)
        cc0 = np.zeros(ntraces)
        cc1 = np.zeros(ntraces)
        err = np.zeros(ntraces)
        tags = np.zeros(ntraces)

        if rank > 0:
            dset.dataset[1] = CCData(stacks, timestamps, fs)

        else:
            pass

        for ix in range(rank, ntraces, size):
            dvvp, dvv_timestp, ccoeffp, \
                best_ccoeffp, dvv_errorp, = dset.measure_dvv_ser(f0=freq_band[0], f1=freq_band[1],
                                                 ref=ref, ngrid=config["ngrid"], stacklevel=1,
                                                 method=config["measurement_type"], indices=[ix],
                                                 dvv_bound=config["maxdvv"],
                                                 )
            t[ix] = dvv_timestp[0]
            dvv[ix] = dvvp[0]
            cc0[ix] = ccoeffp[0]
            cc1[ix] = best_ccoeffp[0]
            err[ix] = dvv_errorp[0]

        comm.barrier()

        # now collect
        if rank == 0:
            t_all = t.copy()
            dvv_all = dvv.copy()
            cc0_all = cc0.copy()
            cc1_all = cc1.copy()
            err_all = err.copy()
            for other_rank in range(1, size):
                comm.Recv(t, source=other_rank, tag=77)
                t_all += t
                comm.Recv(dvv, source=other_rank, tag=78)
                dvv_all += dvv
                comm.Recv(cc0, source=other_rank, tag=79)
                cc0_all += cc0
                comm.Recv(cc1, source=other_rank, tag=80)
                cc1_all += cc1
                comm.Recv(err, source=other_rank, tag=81)
                err_all += err
        else:
            comm.Send(t, dest=0, tag=77)
            comm.Send(dvv, dest=0, tag=78)
            comm.Send(cc0, dest=0, tag=79)
            comm.Send(cc1, dest=0, tag=80)
            comm.Send(err, dest=0, tag=81)

        if rank == 0:
            dvv_list.extend(dvv_all)
            err_list.extend(err_all)
            cc0_list.extend(cc0_all)
            cc1_list.extend(cc1_all)
            t_list.extend(timestamps)
            tags_list.extend([ix_ref for j in range(len(cc0_all))])
        else:
            pass

    if rank == 0:
        return(np.array(t_list), np.array(dvv_list), np.array(cc0_list),
               np.array(cc1_list), np.array(err_list), np.array(tags_list))
    else:
        return([], [], [], [], [], [])

def run_measurement(corrstacks, conf, twin, freq_band, rank, comm):
    output = pd.DataFrame(columns=["timestamps", "t0_s", "t1_s", "f0_Hz",  "f1_Hz",
                                   "tag", "dvv_max", "dvv", "cc_before", "cc_after",
                                   "dvv_err"])
    tags = None
    if conf["reference_type"] == "inversion":
        t, dvv, cc0, cc1, err, tg = measurement_brenguier(corrstacks, conf, twin, freq_band, rank, comm)

    elif conf["reference_type"] in ["list", "bootstrap"]:
        t, dvv, cc0, cc1, err, tags = measurement_list(corrstacks, conf, twin, freq_band, rank, comm)

    elif conf["reference_type"] == "increment":
        t, dvv, cc0, cc1, err, tg = measurement_incremental(corrstacks, conf, twin, freq_band, rank, comm)

    elif conf["reference_type"] == "trailing":
        t, dvv, cc0, cc1, err, tg = measurement_trailing(corrstacks, conf, twin, freq_band, rank, comm)


    if rank == 0:
        for i in range(len(t)):
            if tags is None:
                tag = np.nan
            else:
                tag = tags[i]
            output.loc[i] = [t[i], twin[0], twin[1], freq_band[0], freq_band[1],
                             tag, conf["maxdvv"], dvv[i], cc0[i], cc1[i], err[i]]
        return(output)
    else:
        return(None)