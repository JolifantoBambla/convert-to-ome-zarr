# convert_to_ome_zarr
A small Python module including command line utilities to create volumetric OME-Zarr data sets from other file formats.

## Setup
```
python3 -m venv venv
source venv/bin/activate
pip install .
```

## Commandline usage

This project provides three command line utilities:
 * `ome_zarr_from_imaris`
 * `ome_zarr_from_ome_tiff`
 * `ome_zarr_from_raw`

All three share the following optional arguments:
```
  -h, --help            show this help message and exit
  --chunksize CHUNKSIZE CHUNKSIZE CHUNKSIZE, -c CHUNKSIZE CHUNKSIZE CHUNKSIZE
                        The size of a chunk / brick in the OME-Zarr data set in order 'ZYX'. Defaults to [32, 32, 32]
  --transform TRANSFORM TRANSFORM TRANSFORM, -t TRANSFORM TRANSFORM TRANSFORM
                        The scaling that should be applied to each path in the OME-Zarr data set. Defaults to [1., 1., 1.]
  --interpolator {nearest,linear,gaussian,label_gaussian,bspline,bspline1,bspline2,bspline3,bspline4,bspline5,hamming,cosine,welch,lanczos,blackman,bspline_resampler,bspline_resampler_order1,bspline_resampler_order2,bspline_resampler_order3,bspline_resampler_order4,bspline_resampler_order5}, -i {nearest,linear,gaussian,label_gaussian,bspline,bspline1,bspline2,bspline3,bspline4,bspline5,hamming,cosine,welch,lanczos,blackman,bspline_resampler,bspline_resampler_order1,bspline_resampler_order2,bspline_resampler_order3,bspline_resampler_order4,bspline_resampler_order5}
                        The interpolator that is used to down sample input data sets to create a multi-resolution hierarchy. Defaults to "linear".
  --verbose, --no-verbose, -v
                        Print progress information, etc. Defaults to "True" (default: True)
  --write_separate_channels, --no-write_separate_channels, -w
                        Write channels as separate OME-Zarr files first and combine them afterwards to reduce memory pressure.Defaults to "True". (default: True)
```

### Create OME-Zarr from RAW data set
```
ome_zarr_from_raw 
    [-h]
    [--chunksize CHUNKSIZE CHUNKSIZE CHUNKSIZE]
    [--transform TRANSFORM TRANSFORM TRANSFORM]
    [--interpolator {nearest,linear,gaussian,label_gaussian,bspline,bspline1,bspline2,bspline3,bspline4,bspline5,hamming,cosine,welch,lanczos,blackman,bspline_resampler,bspline_resampler_order1,bspline_resampler_order2,bspline_resampler_order3,bspline_resampler_order4,bspline_resampler_order5}]
    [--verbose | --no-verbose | -v]
    [--write_separate_channels | --no-write_separate_channels | -w]
    [--numduplicates NUMDUPLICATES]
    [--axisorder AXISORDER]
    --size SIZE SIZE SIZE
    --dtype {uint16,uint8,float}
    --outpath OUTPATH
    [files ...]
```

#### Positional Arguments
```
  files                 The list of files to include in the OME-Zarr.
```

#### Required Arguments
```
  --size SIZE_Z SIZE_Y SIZE_X, -s SIZE_Z SIZE_Y SIZE_X
                        The dimensions of the input files in order [depth, height, width].
  --dtype {uint16,uint8,float}, -d {uint16,uint8,float}
                        The data type of the elements in the input files.
  --outpath OUTPATH, -o OUTPATH
                        The file path the OME-Zarr is written to.
```

#### Optional Arguments:
```
  --numduplicates NUMDUPLICATES, -n NUMDUPLICATES
                        Specifies the number of times each input data set is duplicated in the generated OME-Zarr file. This is mainly useful for generating multi-channel datasets from a single channel. Defaults to 1.
  --axisorder AXISORDER, -a AXISORDER
                        The axis order in the input files. Defaults to "ZYX".
```

### Create OME-Zarr from OME-Tiff data set
```
ome_zarr_from_ome_tiff
    [-h]
    [--chunksize CHUNKSIZE CHUNKSIZE CHUNKSIZE]
    [--transform TRANSFORM TRANSFORM TRANSFORM]
    [--interpolator {nearest,linear,gaussian,label_gaussian,bspline,bspline1,bspline2,bspline3,bspline4,bspline5,hamming,cosine,welch,lanczos,blackman,bspline_resampler,bspline_resampler_order1,bspline_resampler_order2,bspline_resampler_order3,bspline_resampler_order4,bspline_resampler_order5}]
    [--verbose | --no-verbose | -v]
    [--write_separate_channels | --no-write_separate_channels | -w]
    --outpath OUTPATH
    file
```

#### Positional Arguments
```
  file                  The OME-Tiff file to convert to OME-Zarr.
```

#### Required Arguments
```
  --outpath OUTPATH, -o OUTPATH
                        The file path the OME-Zarr is written to.
```


### Create OME-Zarr from Imaris data set
```
ome_zarr_from_imaris
    [-h]
    [--chunksize CHUNKSIZE CHUNKSIZE CHUNKSIZE]
    [--transform TRANSFORM TRANSFORM TRANSFORM]
    [--interpolator {nearest,linear,gaussian,label_gaussian,bspline,bspline1,bspline2,bspline3,bspline4,bspline5,hamming,cosine,welch,lanczos,blackman,bspline_resampler,bspline_resampler_order1,bspline_resampler_order2,bspline_resampler_order3,bspline_resampler_order4,bspline_resampler_order5}]
    [--verbose | --no-verbose | -v]
    [--write_separate_channels | --no-write_separate_channels | -w]
    [--resolution_level_lock RESOLUTION_LEVEL_LOCK]
    --outpath OUTPATH
    file
```

#### Positional Arguments
```
  file                  The Imaris file to convert to OME-Zarr.
```

#### Required Arguments
```
  --outpath OUTPATH, -o OUTPATH
                        The file path the OME-Zarr is written to.
```

#### Optional Arguments:
```
  --resolution_level_lock RESOLUTION_LEVEL_LOCK, -r RESOLUTION_LEVEL_LOCK
                        Defines the Imaris file's resolution level to use for generating the OME-Zarr. Defaults to "0".
```

## Example Data for RAW data sets
Example data can be found here:
 - [Open Scientific Visualization Datasets](https://klacansky.com/open-scivis-datasets/)

E.g., for creating an OME-Zarr data set with two channels from the [bunny data set](http://cdn.klacansky.com/open-scivis-datasets/bunny/bunny_512x512x361_uint16.raw) run:
```
    ome_zarr_from_raw \
        -s 361 512 512 \
        -d uint16 \
        -t 0.5 0.337891 0.337891 \
        -n 2 \
        -o /tmp/bunny.zarr\
        bunny_512x512x361_uint16.raw
```
