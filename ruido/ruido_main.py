from ruido.utils.read_config import read_config
from ruido.scripts import measure, cluster, stack
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

configfile = sys.argv[1]
config = read_config(configfile)

if config["do_clustering"]:

    if not config["cluster_by_file"]:
        cluster.run_clustering(config, rank, size, comm)
    else:
        cluster.run_clustering_byfile(config, rank, size, comm)

    if config["print_debug"]:
        print("Rank {} is back from clustering.".format(rank))
    comm.Barrier()

if config["do_stacking"]:
    stack.run_stacking(config, rank, size, comm)
    comm.Barrier()

if config["do_measurement"]:
    measure.run_measure(config, rank, size, comm)
