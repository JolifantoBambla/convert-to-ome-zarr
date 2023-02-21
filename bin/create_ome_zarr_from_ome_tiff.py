#!/usr/bin/env python

import argparse
import numpy as np
from create_ome_zarr import create_ome_zarr, INTERPOLATOR_MAPPING, OmeTiffDataSource


if __name__ == '__main__':
    # todo: Allow more dtypes #2
    dtype_mapping = {
        'uint16': np.uint16,
        'uint8': np.uint8,
        'float': float
    }

    parser = argparse.ArgumentParser(
        description="Convert .raw files to the OME-Zarr format. If more than one file is given, each file will be "
                    "stored as a separate channel in the OME-Zarr data set. All files must have the same dimension. "
    )
    parser.add_argument(
        "--chunksize",
        "-c",
        type=int,
        nargs=3,
        default=[32, 32, 32],
        help="The size of a chunk / brick in the OME-Zarr data set. Defaults to [32, 32, 32]"
    )
    parser.add_argument(
        "--transform",
        "-t",
        type=float,
        nargs=3,
        default=None,
        help="The scaling that should be applied to each path in the OME-Zarr data set. Defaults to [1., 1., 1.]"
    )
    parser.add_argument(
        "--interpolator",
        "-i",
        type=str,
        choices=list(INTERPOLATOR_MAPPING.keys()),
        default="linear",
        help="The interpolator that is used to down sample input data sets to create a multi-resolution hierarchy. "
             "Defaults to \"linear\"."
    )

    # todo: Create omero meta data from OME-XML or from OME-Tiff meta data #1
    #parser.add_argument(
    #    "--omexml",
    #    "-x",
    #    type=str,
    #    help="An optional OME-XML file containing meta data. It is used to generate an \"omero\" meta data block in the"
    #         "OME-Zarr."
    #)

    parser.add_argument(
        "--verbose",
        "-v",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print progress information, etc. Defaults to \"True\""
    )
    parser.add_argument(
        "--write_separate_channels",
        "-w",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write channels as separate OME-Zarr files first and combine them afterwards to reduce memory pressure."
             "Defaults to \"True\"."
    )
    parser.add_argument(
        "--outpath",
        "-o",
        type=str,
        required=True,
        help="The file path the OME-Zarr gets be written to."
    )
    parser.add_argument(
        "file",
        type=str,
        nargs=1,
        help="The list of files to include in the OME-Zarr."
    )
    args = parser.parse_args()

    transformations = None
    if args.transform is not None:
        transform = args.transform
        transform.insert(0, 1.)
        transform.insert(0, 1.)
        transformations = [dict(type="scale", scale=transform)]

    data_source = OmeTiffDataSource(args.file)

    create_ome_zarr(data_source,
                    args.outpath,
                    chunk_shape=args.chunksize,
                    interpolator=INTERPOLATOR_MAPPING[args.interpolator],
                    coordinate_transformations=transformations,
                    write_channels_as_separate_files=args.write_separate_channels,
                    verbose=args.verbose)
