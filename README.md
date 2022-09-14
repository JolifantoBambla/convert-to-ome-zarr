# raw-to-ome-zarr
Create OME-Zarr data sets from raw volumetric data.

## Setup
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Commandline usage
```
create_ome_zarr_from_raw.py
    [-h]
    [--chunksize CHUNKSIZE CHUNKSIZE CHUNKSIZE]
    [--interpolator {nearest,linear,gaussian,label_gaussian,bspline,bspline1,bspline2,bspline3,bspline4,bspline5,hamming,cosine,welch,lanczos,blackman,bspline_resampler,bspline_resampler_order1,bspline_resampler_order2,bspline_resampler_order3,bspline_resampler_order4,bspline_resampler_order5}]
    [--numduplicates NUMDUPLICATES]
    [--axisorder AXISORDER]
    --size SIZE SIZE SIZE
    --dtype {uint16,uint8,float}
    --outpath OUTPATH
    [files ...]
```

### Positional Arguments
```
  files                 The list of files to include in the OME-Zarr.
```

### Required Arguments
```
  --size SIZE_Z SIZE_Y SIZE_X, -s SIZE_Z SIZE_Y SIZE_X
                        The dimensions of the input files in order [depth, height, width].
  --dtype {uint16,uint8,float}, -d {uint16,uint8,float}
                        The data type of the elements in the input files.
  --outpath OUTPATH, -o OUTPATH
                        The file path the OME-Zarr gets be written to.
```

### Optional Arguments:
```
  -h, --help            show this help message and exit
  --chunksize CHUNKSIZE_Z CHUNKSIZE_Y CHUNKSIZE_X, -c CHUNKSIZE_Z CHUNKSIZE_Y CHUNKSIZE_X
                        The size of a chunk / brick in the OME-Zarr data set. Defaults to [32, 32, 32]
  --interpolator {nearest,linear,gaussian,label_gaussian,bspline,bspline1,bspline2,bspline3,bspline4,bspline5,hamming,cosine,welch,lanczos,blackman,bspline_resampler,bspline_resampler_order1,bspline_resampler_order2,bspline_resampler_order3,bspline_resampler_order4,bspline_resampler_order5}, -i {nearest,linear,gaussian,label_gaussian,bspline,bspline1,bspline2,bspline3,bspline4,bspline5,hamming,cosine,welch,lanczos,blackman,bspline_resampler,bspline_resampler_order1,bspline_resampler_order2,bspline_resampler_order3,bspline_resampler_order4,bspline_resampler_order5}
                        The interpolator that is used to down sample input data sets to create a multi-resolution hierarchy. Defaults to "linear".
  --numduplicates NUMDUPLICATES, -n NUMDUPLICATES
                        Specifies the number of times each input data set is duplicated in the generated OME-Zarr file. This is mainly useful for generating multi-channel datasets from a single channel. Defaults to 1.
  --axisorder AXISORDER, -a AXISORDER
                        The axis order in the input files. Defaults to "ZYX".
```

## Example Data
Example data can be found here:
 - [Open Scientific Visualization Datasets](https://klacansky.com/open-scivis-datasets/)
