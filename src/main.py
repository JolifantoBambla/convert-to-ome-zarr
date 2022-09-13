### https://klacansky.com/open-scivis-datasets/
### All datasets are in little-endian byte order. Dimensions are width x height x depth (e.g., array[depth][height][width] in C).
import tempfile
from dataclasses import dataclass
from os import PathLike
from typing import List, Tuple, Union

import argparse
import csv
import functools
import itertools
import multiprocessing
import os
import sys

import SimpleITK as sitk

import numpy as np
from numpy.typing import DTypeLike, NDArray
import tifffile as tf

import zarr

import ome_zarr
from ome_zarr.io import parse_url
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_multiscale


ShapeLike3d = Union[List[int], Tuple[int, int, int]]
ShapeLike5d = Union[List[int], Tuple[int, int, int, int, int]]


@dataclass
class RawFileInfo:
    path: Union[str, bytes, PathLike]
    shape: Tuple[int, int, int]
    data_type: DTypeLike


def read_raw(file_name: Union[str, bytes, PathLike], shape: ShapeLike5d, dtype: DTypeLike) -> NDArray:
    arr = np.fromfile(file_name, dtype=dtype)
    return arr.reshape(shape)


def file_infos_are_valid(file_infos: List[RawFileInfo]) -> bool:
    allowed_dtypes = [np.uint8, np.uint16]

    base_shape = file_infos[0].shape
    data_type = file_infos[0].data_type

    return all([f.shape is base_shape and f.data_type is data_type for f in file_infos]) and data_type in allowed_dtypes


def read_files(file_infos: List[RawFileInfo]) -> List[NDArray]:
    if not file_infos_are_valid(file_infos):
        raise RuntimeError("File infos don't have the same shape and/or data type.")
    return list(map(lambda f: read_raw(f.path, f.shape, f.data_type), file_infos))


def write_as_ome_zarr(out_path: Union[str, bytes, PathLike], pyramid: List[NDArray], chunk_shape: ShapeLike5d):
    """

    :param out_path:
    :param pyramid:
    :param chunk_shape: (depth, height, width)
    """

    kwargs = {}

    store = parse_url(out_path, mode="w").store
    root = zarr.group(store=store)
    write_multiscale(
        pyramid=pyramid,
        group=root,
        axes="tczyx",
        chunks=chunk_shape,
        # according to the docs this is the recommended way to specify the chunk shape now, but it has no effect...
        storage_options=dict(chunks=chunk_shape),
    )
    #root.attrs["omero"] = {
    #    "channels": [{
    #        "color": "00FFFF",
    #        "window": {"start": 0, "end": 20},
    #        "label": "random",
    #        "active": True,
    #    }]
    #}


# https://discourse.itk.org/t/resample-volume-to-specific-voxel-spacing-simpleitk/3531
def resample_volume(volume: sitk.Image, interpolator=sitk.sitkLinear, new_spacing=None) -> sitk.Image:
    if new_spacing is None:
        new_spacing = [2.0, 2.0, 2.0]
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    return sitk.Resample(
        volume,
        new_size,
        sitk.Transform(),
        interpolator,
        volume.GetOrigin(),
        new_spacing,
        volume.GetDirection(),
        0,
        volume.GetPixelID()
    )


def sitk_to_numpy(volume: sitk.Image) -> NDArray:
    return sitk.GetArrayFromImage(volume)


def main(argv=None):
    pass


def compute_multiscale_3d(input_shape: ShapeLike3d, target_shape: ShapeLike3d) -> List[ShapeLike3d]:
    last_shape = input_shape
    scales = [[1., 1., 1.]]

    while any([last_shape[i] > target_shape[i] for i in range(3)]):
        last_scale = scales[-1:][0]
        scale = [s * 2. if all([last_shape[i] >= last_shape[j] / 2. for j in range(3) if i != j]) else 1. for i, s in enumerate(last_scale)]
        scales.append(scale)
        last_shape = [n / s for n, s in zip(input_shape, scale)]

    return scales


if __name__ == '__main__':
    # For raw image (without any headers)
    in_path = "/home/ripley/Documents/data/raw/bunny_512x512x361_uint16.raw"
    dtype = np.uint16
    shape = 1, 1, 361, 512, 512,

    out_path = "/tmp/bunny.zarr"
    chunk_shape = 1, 1, 32, 32, 32,

    #write_as_ome_zarr(
    #    out_path,
    #    read_files([RawFileInfo(in_path, shape, dtype)]),
    #    tile_shape
    #)

    input_shape_3d = shape[2:]
    multiscale = compute_multiscale_3d(input_shape_3d, chunk_shape[2:])

    numpy_volume = read_raw(in_path, input_shape_3d, dtype)

    print(np.shape(numpy_volume))

    volume = sitk.GetImageFromArray(numpy_volume.reshape(input_shape_3d))
    print(np.shape(sitk_to_numpy(volume)))

    pyramid = [numpy_volume.reshape(shape)]

    for s in multiscale:
        scale = s.copy()
        scale.reverse()

        resampled = sitk_to_numpy(resample_volume(volume, new_spacing=scale))

        resolution_shape = list(resampled.shape)
        resolution_shape.insert(0, 1)
        resolution_shape.insert(0, 1)

        pyramid.append(resampled.reshape(resolution_shape))

    # todo: ome_zarr.write_multiscales
    write_as_ome_zarr(
        out_path,
        pyramid,
        chunk_shape
    )

    #sys.exit(main())
