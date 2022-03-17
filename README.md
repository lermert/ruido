### Ruido - a set of objects to handle highly time-resolved ambient noise correlation data

The purpose of these objects and scripts is mainly to handle sets of short-term ambient noise correlation observations. When we compute hourly or sub-hourly correlations from continuous ambient noise data, we end up with tens of thousands of individual windows. This module is designed to work on such data contained in hdf5 files with the format that is shown below (hdf5 file format). This format is written out directly by the ants module (https://github.com/lermert/ants_2).


### Idea

There is a serial and a parallel version designed to handle the same tasks, but one in a serial context and one using mpi4py to split processing tasks among processes.

The central part is the CCDataSet object which itself is a dictionary of CCData objects. The most important tasks these fulfill is reading the data from hdf5 (either partially or full), tapering / filtering them, plotting them, selecting them for stacking based on various criteria, and forming the stack, as well as taking stretching measurements with respect to a reference.

Head to https://github.com/lermert/cdmx_dvv where they are put into use. This module is just a bookkeeping device.


#### hdf5 file format

HDF5 "NET.STA.LOCATION.CHANNEL--NET.STA.LOCATION.CHANNEL.ccc.windows.h5" {
GROUP "/" {
   GROUP "corr_windows" {
      DATASET "data" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SIMPLE { ( NR OF TRACES, NR OF SAMPLES ) / ( NR OF TRACES, NR OF SAMPLES ) }
      }
      DATASET "timestamps" {
         DATATYPE  H5T_STRING {
            STRSIZE H5T_VARIABLE;
            STRPAD H5T_STR_NULLTERM;
            CSET H5T_CSET_UTF8;
            CTYPE H5T_C_S1;
         }
         DATASPACE  SIMPLE { ( NR OF TRACES ) / ( NR OF SAMPLES ) }
      }
   }
   DATASET "stats" {
      DATATYPE  H5T_STD_I64LE
      DATASPACE  SIMPLE { ( 1 ) / ( 1 ) }
      ATTRIBUTE "channel1" {
         DATATYPE  H5T_STRING {
            STRSIZE H5T_VARIABLE;
            STRPAD H5T_STR_NULLTERM;
            CSET H5T_CSET_UTF8;
            CTYPE H5T_C_S1;
         }
         DATASPACE  SCALAR
      }
      ATTRIBUTE "channel2" {
         DATATYPE  H5T_STRING {
            STRSIZE H5T_VARIABLE;
            STRPAD H5T_STR_NULLTERM;
            CSET H5T_CSET_UTF8;
            CTYPE H5T_C_S1;
         }
         DATASPACE  SCALAR
      }
      ATTRIBUTE "distance" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SCALAR
      }
      ATTRIBUTE "sampling_rate" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SCALAR
      }
   }
}
}
