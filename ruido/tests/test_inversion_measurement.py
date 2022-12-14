# coding: utf-8
from ruido.scripts.measurements import measurement_brenguier
from ruido.classes.cc_dataset_mpi import CCDataset
from ruido.utils.noisepy import stretching_vect
import matplotlib.pyplot as plt
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

d = CCDataset("tests/testdata_stretching.h5")
d.data_to_memory()
d.dataset[1] = d.dataset[0]


# para = {}
# para["dt"] = 0.01
# para["twin"] = [0., 100.]  # [self.lag[0], self.lag[-1] + 1. / self.fs]
# para["freq"] = [3.9, 4.1]

# a = stretching_vect(ref=d.dataset[0].data[0], cur=d.dataset[0].data[1],
#                     dv_range=0.05, nbtrial=100, para=para)
# print(a)

# a = stretching_vect(ref=d.dataset[0].data[0], cur=d.dataset[0].data[2],
#                     dv_range=0.05, nbtrial=100, para=para)
# print(a)

conf = {}
conf["badwins"] = []
conf["ngrid"] = 300
conf["measurement_type"] = "stretching"
conf["maxdvv"] = 0.05
conf["brenguier_beta"] = 3


a = measurement_brenguier(d, conf, [0, 100.], [3.9, 4.1], rank, comm)

print(a[1])
plt.plot(a[0], a[1])
plt.grid()
plt.show()
