import argparse
from os import PathLike
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike, NDArray
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale
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

    return np.moveaxis(arr, [target_axes.index(axis_name) for axis_name in input_axes], np.arange(len(target_axes)))


def write_as_ome_zarr(out_path: FilePath, pyramid: List[NDArray], chunk_shape: ShapeLike5d):
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


def compute_multiscale_3d(input_shape: ShapeLike3d, target_shape: ShapeLike3d) -> List[ShapeLike3d]:
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
        volume = move_axes(read_raw(f, shape_5d, dtype), f'TC{axis_order}')
        volume_sitk = sitk.GetImageFromArray(volume.reshape(shape))

        if i == 0:
            pyramid.append(volume)
        else:
            pyramid[0] = np.append(pyramid[0], volume, axis=1)

        for j, s in enumerate(multiscale):
            scale = s.copy()
            scale.reverse()
            resampled = sitk.GetArrayFromImage(resample_volume(volume_sitk, new_spacing=scale, interpolator=interpolator))
            resampled = resampled.reshape(convert_5d_shape(resampled.shape))

            if i == 0:
                pyramid.append(resampled)
            else:
                level = j + 1
                pyramid[level] = np.append(pyramid[level], resampled, axis=1)

    write_as_ome_zarr(out_path, pyramid, chunk_shape_5d)


if __name__ == '__main__':
    # todo: allow more dtypes
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
        "--interpolator",
        "-i",
        type=str,
        choices=list(interpolator_mapping.keys()),
        default="linear",
        help="The interpolator that is used to down sample input data sets to create a multi-resolution hierarchy. "
             "Defaults to \"linear\"."
    )
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

    create_ome_zarr_from_raw([f for f in args.files for _ in range(args.numduplicates)],
                             args.size,
                             dtype_mapping[args.dtype],
                             args.outpath,
                             chunk_shape=args.chunksize,
                             interpolator=interpolator_mapping[args.interpolator])
