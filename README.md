### Ruido - a set of objects to handle highly time-resolved ambient noise correlation data

The purpose of these objects and scripts is mainly to handle sets of short-term ambient noise correlation observations, and bring us from the (short-term unstacked) correlation data to .csv files with stretching measurements.

When we compute hourly or sub-hourly correlations from continuous ambient noise data, we end up with tens of thousands of individual windows. This module is designed to work on such data contained in hdf5 files with the format that is shown below (hdf5 file format). Correlations in this format are written out directly by the ants module (https://github.com/lermert/ants_2).


### Idea

Scripts that do stuff are in the scripts directory. One is for clustering following Viens & Iwata (2020), one for stacking, and one for taking dv/v measurements. 

To use the scripts, take these steps: 
1. Edit the config.yml file. Set do_clustering, do_stacking and do_measurement to true or false. Then, edit all the rest. Do not remove any parameters, just ignore the irrelevant ones (e.g. when do_clustering is false, all parameters relevant to clustering will simply be ignored.)
2. Call python <path-to-ruido-main>/ruido_main.py <configfile-name>
3. For parallel run, call mpirun -np <nr-processes> <path-to-ruido-main>/ruido_main.py <configfile-name>

You can also use the objects in classes/ interactively to load, filter, stack correlation data.

### Underlying idea

There is a serial and a parallel version designed to handle the same tasks, but one in a serial context and one using mpi4py to split processing tasks among processes.

The central part is the CCDataSet object which itself is a dictionary of CCData objects. The most important tasks these fulfill is reading the data from hdf5 (either partially or full), tapering / filtering them, plotting them, selecting them for stacking based on various criteria, and forming the stack, as well as taking stretching measurements with respect to a reference.


#### hdf5 file format for data (single floats could be used in place of doubles, too)

HDF5 "NET.STA.LOCATION.CHANNEL--NET.STA.LOCATION.CHANNEL.ccc.windows.h5" {\
GROUP "/" {\
   GROUP "corr_windows" {\
      DATASET "data" {\
         DATATYPE  H5T_IEEE_F64LE\
         DATASPACE  SIMPLE { ( NR OF TRACES, NR OF SAMPLES ) / ( NR OF TRACES, NR OF SAMPLES ) }\
      }\
      DATASET "timestamps" {\
         DATATYPE  H5T_STRING {\
            STRSIZE H5T_VARIABLE;\
            STRPAD H5T_STR_NULLTERM;\
            CSET H5T_CSET_UTF8;\
            CTYPE H5T_C_S1;\
         }\
         DATASPACE  SIMPLE { ( NR OF TRACES ) / ( NR OF TRACES ) }\
      }\
   }\
   DATASET "stats" {\
      DATATYPE  H5T_STD_I64LE\
      DATASPACE  SIMPLE { ( 1 ) / ( 1 ) }\
      ATTRIBUTE "channel1" {\
         DATATYPE  H5T_STRING {\
            STRSIZE H5T_VARIABLE;\
            STRPAD H5T_STR_NULLTERM;\
            CSET H5T_CSET_UTF8;\
            CTYPE H5T_C_S1;\
         }\
         DATASPACE  SCALAR\
      }\
      ATTRIBUTE "channel2" {\
         DATATYPE  H5T_STRING {\
            STRSIZE H5T_VARIABLE;\
            STRPAD H5T_STR_NULLTERM;\
            CSET H5T_CSET_UTF8;\
            CTYPE H5T_C_S1;\
         }\
         DATASPACE  SCALAR\
      }\
      ATTRIBUTE "distance" {\
         DATATYPE  H5T_IEEE_F64LE\
         DATASPACE  SCALAR\
      }\
      ATTRIBUTE "sampling_rate" {\
         DATATYPE  H5T_IEEE_F64LE\
         DATASPACE  SCALAR\
      }\
   }\
}\
}
