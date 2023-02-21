import argparse
from os import PathLike
import os
import shutil
from typing import List, Tuple, Union, Dict, Any

import hdf5plugin  # needed for reading ims files (no idea why they wouldn't just import it in 'imaris_ims_file_reader')
import imaris_ims_file_reader.ims as ims
import json
import numpy as np
from numpy.typing import DTypeLike, NDArray
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale
import tifffile as tf
import SimpleITK as sitk
import zarr

FilePath = Union[str, bytes, PathLike]
ShapeLike3d = Union[List[int], Tuple[int, int, int]]
ShapeLike5d = Union[List[int], Tuple[int, int, int, int, int]]


def read_raw(file_name: FilePath, shape: ShapeLike5d, dtype: DTypeLike) -> NDArray:
    arr = np.fromfile(file_name, dtype=dtype)
    return arr.reshape(shape)


def move_axes(arr: NDArray, input_axes: str, target_axes='STCZYX') -> NDArray:
    input_axes = list(input_axes.upper())
    target_axes = list(target_axes.upper())

    if len(input_axes) != len(target_axes):
        target_axes = [axis_name for axis_name in target_axes if axis_name in input_axes]
        assert len(input_axes) == len(target_axes), 'Incompatible input and target axes names: input={}; target={}'.format(input_axes, target_axes)

    if input_axes == target_axes:
        return arr

    data_shape = list(arr.shape)
    if len(data_shape) < len(input_axes):
        for i in range(len(input_axes) - len(arr.shape)):
            data_shape.insert(0, 1)

    return np.moveaxis(
        arr if len(data_shape) == len(arr.shape) else arr.reshape(data_shape),
        [target_axes.index(axis_name) for axis_name in input_axes],
        np.arange(len(target_axes))
    )


def write_as_ome_zarr(out_path: FilePath, pyramid: List[NDArray], chunk_shape: ShapeLike5d,
                      coordinate_transformations: List[List[Dict[str, Any]]] = None):
    """

    :param out_path:
    :param pyramid:
    :param chunk_shape: (depth, height, width)
    :param coordinate_transformations:
    """

    kwargs = {}

    store = parse_url(out_path, mode="w").store
    root = zarr.group(store=store)

    # todo: Create omero meta data from OME-XML or from OME-Tiff meta data #1
    #root.attrs["omero"] = {
    #    "channels": [{
    #        "color": "00FFFF",
    #        "window": {"start": 0, "end": 20},
    #        "label": "random",
    #        "active": True,
    #    }]
    #}

    write_multiscale(
        pyramid=pyramid,
        group=root,
        axes="tczyx",
        chunks=chunk_shape,
        # according to the docs this is the recommended way to specify the chunk shape now, but it has no effect...
        storage_options=dict(chunks=chunk_shape),
        coordinateTransformations=coordinate_transformations,
    )


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


def compute_multiscale_3d(input_shape: ShapeLike3d, target_shape: ShapeLike3d,) -> List[ShapeLike3d]:
    last_shape = input_shape
    scales = [[1., 1., 1.]]

    while any([last_shape[i] > target_shape[i] for i in range(3)]):
        last_scale = scales[-1:][0]
        scale = [s * 2. if all([last_shape[i] >= last_shape[j] / 2. for j in range(3) if i != j]) else 1. for i, s in enumerate(last_scale)]
        scales.append(scale)
        last_shape = [n / s for n, s in zip(input_shape, scale)]

    return scales


def create_ome_zarr_from_raw(files: List[FilePath], shape: ShapeLike3d, dtype: DTypeLike, out_path: FilePath,
                             chunk_shape: ShapeLike3d = None,
                             coordinate_transformations: List[List[Dict[str, Any]]] = None,
                             axis_order='ZYX',
                             interpolator=sitk.sitkLinear):
    assert len(axis_order) == 3, f'Expected axis order to have length 3, got {axis_order}'
    assert all([a in axis_order.upper() for a in 'ZYX']), f'Expected axis order to contain X, Y, and Z, got {axis_order.upper()}'
    if chunk_shape is None:
        chunk_shape = [32, 32, 32]

    def convert_5d_shape(shape_3d: ShapeLike3d) -> ShapeLike5d:
        s5d = list(shape_3d)
        s5d.insert(0, 1)
        s5d.insert(0, 1)
        return s5d

    shape_5d = convert_5d_shape(shape)
    chunk_shape_5d = convert_5d_shape(chunk_shape)

    multiscale = compute_multiscale_3d(shape, chunk_shape)
    pyramid = []
    for i, f in enumerate(files):
        # todo: Allow larger data sets #3
        volume = move_axes(read_raw(f, shape_5d, dtype), f'TC{axis_order}')
        volume_sitk = sitk.GetImageFromArray(volume.reshape(shape))

        if i == 0:
            pyramid.append(volume)
        else:
            pyramid[0] = np.append(pyramid[0], volume, axis=1)

        for j, s in enumerate(multiscale[1:]):
            scale = s.copy()
            scale.reverse()
            resampled = sitk.GetArrayFromImage(resample_volume(volume_sitk, new_spacing=scale, interpolator=interpolator))
            resampled = resampled.reshape(convert_5d_shape(resampled.shape))

            if i == 0:
                pyramid.append(resampled)
            else:
                level = j + 1
                pyramid[level] = np.append(pyramid[level], resampled, axis=1)

    write_as_ome_zarr(out_path, pyramid, chunk_shape_5d, coordinate_transformations=coordinate_transformations)


def combine_ome_zarr_channels(out_path: FilePath, num_scales: int, target_shape: ShapeLike5d):
    os.replace(f'{out_path}_channel_0', f'{out_path}')

    for s in range(num_scales):
        with open(f'{out_path}/{s}/.zarray', 'r') as zarray_file:
            meta = json.load(zarray_file)

        meta['shape'][1] = target_shape[1]
        with open(f'{out_path}/{s}/.zarray', 'w') as zarray_file:
            json.dump(meta, zarray_file, sort_keys=True, indent=4)

    for c in range(1, target_shape[1]):
        for s in range(num_scales):
            shutil.move(f'{out_path}_channel_{c}/{s}/0/0', f'{out_path}/{s}/0/{c}')
        shutil.rmtree(f'{out_path}_channel_{c}')


def create_ome_zarr_from_imaris(file: FilePath, out_path: FilePath,
                                write_channels_as_separate_files=True,
                                chunk_shape: ShapeLike3d = None,
                                coordinate_transformations: List[List[Dict[str, Any]]] = None,
                                interpolator=sitk.sitkLinear):
    resolution_level_lock = 0
    chunk_shape = [128, 128, 128]

    ims_file = ims(file, ResolutionLevelLock=resolution_level_lock)
    shape_5d = ims_file.shape
    chunk_shape_5d = [1, 1] + list(chunk_shape)

    multiscale = compute_multiscale_3d(shape_5d[2:], chunk_shape)
    pyramid = []
    print('channels', shape_5d[1], shape_5d[2:])
    for i in range(shape_5d[1]):
        print('begin channel', i)
        if write_channels_as_separate_files:
            pyramid = []

        volume = np.array(ims_file.hf['DataSet'][f'ResolutionLevel {resolution_level_lock}']['TimePoint 0'][f'Channel {i}']['Data'])
        volume_sitk = sitk.GetImageFromArray(volume)
        volume = volume.reshape([1, 1] + list(volume.shape))

        if i == 0 or write_channels_as_separate_files:
            pyramid.append(volume)
        else:
            pyramid[0] = np.append(pyramid[0], volume, axis=1)

        for j, s in enumerate(multiscale[1:]):
            scale = s.copy()
            scale.reverse()
            resampled = sitk.GetArrayFromImage(
                resample_volume(volume_sitk, new_spacing=scale, interpolator=interpolator))
            resampled = resampled.reshape([1, 1] + list(resampled.shape))

            if i == 0 or write_channels_as_separate_files:
                pyramid.append(resampled)
            else:
                level = j + 1
                pyramid[level] = np.append(pyramid[level], resampled, axis=1)

        if write_channels_as_separate_files:
            write_as_ome_zarr(f'{out_path}_channel_{i}', pyramid, chunk_shape_5d, coordinate_transformations=coordinate_transformations)
        print('finish channel', i)

    if write_channels_as_separate_files:
        combine_ome_zarr_channels(out_path, len(multiscale), shape_5d)
    else:
        write_as_ome_zarr(out_path, pyramid, chunk_shape_5d, coordinate_transformations=coordinate_transformations)


# todo: this is basically the same as create_ome_zarr_from_raw -> refactor
def create_ome_zarr_from_ome_tiff(file: FilePath, out_path: FilePath,
                                  chunk_shape: ShapeLike3d = None,
                                  coordinate_transformations: List[List[Dict[str, Any]]] = None,
                                  axis_order='CZYX',
                                  interpolator=sitk.sitkLinear):
    # todo: most (all?) meta data can be read from the tiff itself
    #tiff = tf.TiffFile(file)
    #axis_order = 'CZYX'[:4-len(tiff.series[0].axes)] + tiff.series[0].axes.upper()
    #print(axis_order, tiff.series[0].shape, tiff.pages[0].shape, len(tiff.pages), 105*15, len(tiff.series))

    data = move_axes(tf.imread(file), 'CZYX'[:4-len(axis_order)] + axis_order.upper())

    shape_4d = data.shape
    shape_3d = shape_4d[1:]
    shape_5d = [1] + list(shape_4d)
    chunk_shape_5d = [1, 1] + list(chunk_shape)

    multiscale = compute_multiscale_3d(shape_3d, chunk_shape)
    pyramid = [data.reshape(shape_5d)]
    for i in range(shape_4d[0]):
        volume = data[i]
        volume_sitk = sitk.GetImageFromArray(volume)

        for j, s in enumerate(multiscale[1:]):
            scale = s.copy()
            scale.reverse()
            resampled = sitk.GetArrayFromImage(
                resample_volume(volume_sitk, new_spacing=scale, interpolator=interpolator))
            resampled = resampled.reshape([1, 1] + list(resampled.shape))

            if i == 0:
                pyramid.append(resampled)
            else:
                level = j + 1
                pyramid[level] = np.append(pyramid[level], resampled, axis=1)

    write_as_ome_zarr(out_path, pyramid, chunk_shape_5d, coordinate_transformations=coordinate_transformations)


if __name__ == '__main__':
    # todo: Allow more dtypes #2
    dtype_mapping = {
        'uint16': np.uint16,
        'uint8': np.uint8,
        'float': float
    }

    interpolator_mapping = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'gaussian': sitk.sitkGaussian,
        'label_gaussian': sitk.sitkLabelGaussian,
        'bspline': sitk.sitkBSpline,
        'bspline1': sitk.sitkBSpline1,
        'bspline2': sitk.sitkBSpline2,
        'bspline3': sitk.sitkBSpline3,
        'bspline4': sitk.sitkBSpline4,
        'bspline5': sitk.sitkBSpline5,
        'hamming': sitk.sitkHammingWindowedSinc,
        'cosine': sitk.sitkCosineWindowedSinc,
        'welch': sitk.sitkWelchWindowedSinc,
        'lanczos': sitk.sitkLanczosWindowedSinc,
        'blackman': sitk.sitkBlackmanWindowedSinc,
        'bspline_resampler': sitk.sitkBSplineResampler,
        'bspline_resampler_order1': sitk.sitkBSplineResamplerOrder1,
        'bspline_resampler_order2': sitk.sitkBSplineResamplerOrder2,
        'bspline_resampler_order3': sitk.sitkBSplineResamplerOrder3,
        'bspline_resampler_order4': sitk.sitkBSplineResamplerOrder4,
        'bspline_resampler_order5': sitk.sitkBSplineResamplerOrder5,
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
        choices=list(interpolator_mapping.keys()),
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
        "--numduplicates",
        "-n",
        type=int,
        default=1,
        help="Specifies the number of times each input data set is duplicated in the generated OME-Zarr file. This is "
             "mainly useful for generating multi-channel datasets from a single channel. Defaults to 1. "
    )
    parser.add_argument(
        "--axisorder",
        "-a",
        type=str,
        default="ZYX",
        help="The axis order in the input files. Defaults to \"ZYX\"."
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        nargs=3,
        required=True,
        help="The dimensions of the input files in order [depth, height, width]."
    )
    parser.add_argument(
        "--dtype",
        "-d",
        required=True,
        choices=list(dtype_mapping.keys()),
        help="The data type of the elements in the input files."
    )
    parser.add_argument(
        "--outpath",
        "-o",
        type=str,
        required=True,
        help="The file path the OME-Zarr gets be written to."
    )
    parser.add_argument(
        "files",
        type=str,
        nargs='+',
        help="The list of files to include in the OME-Zarr."
    )
    args = parser.parse_args()

    transformations = None
    if args.transform is not None:
        transform = args.transform
        transform.insert(0, 1.)
        transform.insert(0, 1.)
        transformations = [dict(type="scale", scale=transform)]


    # todo: split into separate files
    if args.files[0].endswith('tiff') or args.files[0].endswith('tif'):
        create_ome_zarr_from_ome_tiff(args.files[0],
                                      args.outpath,
                                      axis_order='ZCYX',
                                      chunk_shape=args.chunksize,
                                      interpolator=interpolator_mapping[args.interpolator],
                                      coordinate_transformations=transformations)
    elif args.files[0].endswith('ims'):
        create_ome_zarr_from_imaris(args.files[0],
                                    args.outpath,
                                    chunk_shape=args.chunksize,
                                    interpolator=interpolator_mapping[args.interpolator],
                                    coordinate_transformations=transformations)
    else:
        create_ome_zarr_from_raw([f for f in args.files for _ in range(args.numduplicates)],
                                 args.size,
                                 dtype_mapping[args.dtype],
                                 args.outpath,
                                 chunk_shape=args.chunksize,
                                 interpolator=interpolator_mapping[args.interpolator],
                                 coordinate_transformations=transformations)
