from abc import ABC, abstractmethod
import gc
from math import floor
from os import PathLike
import os
import shutil
from tempfile import mkdtemp
from typing import List, Tuple, Union, Dict, Any

import dask
import dask.array as da
import dask_image.imread
import hdf5plugin  # needed for reading ims files (unfortunately, it is not imported in 'imaris_ims_file_reader')
import imaris_ims_file_reader.ims as ims
import json
import numpy as np
from numpy.typing import DTypeLike, NDArray
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale
import tifffile as tf
from scipy import ndimage
import SimpleITK as sitk
import zarr


FilePath = Union[str, bytes, PathLike]
ShapeLike3d = Union[List[int], Tuple[int, int, int]]
ShapeLike5d = Union[List[int], Tuple[int, int, int, int, int]]

INTERPOLATOR_MAPPING = {
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


_DASK_CHUNK_SHAPE = (512, 512, 512)


def read_raw(file_name: FilePath, shape: ShapeLike5d, dtype: DTypeLike) -> da.Array:
    arr = da.from_array(np.fromfile(file_name, dtype=dtype))
    return arr.reshape(shape).rechunk(chunks=_DASK_CHUNK_SHAPE)


def move_axes(arr: NDArray | da.Array, input_axes: str, target_axes='STCZYX') -> NDArray | da.Array:
    """
    Rearranges the axes of a given array such that they are ordered based on a given ordering.
    If the order of the axes in the input array matches the desired order, this is a no-op.

    @param arr: the array
    @param input_axes: the order of the axes in the input array
    @param target_axes: the desired order of the axes
    @return: the rearranged array
    """
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


def compute_multiscale_3d(input_shape: ShapeLike3d, target_shape: ShapeLike3d,) -> List[ShapeLike3d]:
    """
    Computes a sequence of scaling parameters to successively downscale a given input shape to a given target shape,
    where all(input_shape >= target_shape) is true.
    At each step the resolution in each dimension is either halved or stays the same, depending on the other dimensions.

    @param input_shape: the input shape to scale down to a target shape
    @param target_shape: the target shape the input shape should be scaled down to
    @return: a sequence of scaling parameters
    """
    last_shape = input_shape
    scales = [[1., 1., 1.]]

    while any([last_shape[i] > target_shape[i] for i in range(3)]):
        last_scale = scales[-1:][0]
        scale = [s * 2. if all([last_shape[i] >= last_shape[j] / 2. for j in range(3) if i != j]) else 1. for i, s in enumerate(last_scale)]
        scales.append(scale)
        last_shape = [n / s for n, s in zip(input_shape, scale)]

    return scales


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


def write_ome_zarr(out_path: FilePath,
                   pyramid: List[NDArray | da.Array],
                   chunk_shape: ShapeLike5d,
                   coordinate_transformations: List[List[Dict[str, Any]]] = None):
    """
    Writes a given sequence of numpy arrays as an OME-Zarr file to a given path.

    Note: Currently, there is no way to add additional meta-data to the OME-Zarr.

    @param out_path: the path to write the OME-Zarr to.
    @param pyramid: a sequence of numpy arrays.
    @param chunk_shape: the size of a chunk in the resulting OME-Zarr
    @param coordinate_transformations: a series of coordinate transformations to write to the OME-Zarr
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
        storage_options=dict(
            chunks=chunk_shape,
            dimension_separator='/'
        ),
        coordinateTransformations=coordinate_transformations,
    )


def combine_ome_zarr_channels(out_path: FilePath, num_scales: int, target_shape: ShapeLike5d, remove_individual_files=True):
    """
    Combines multiple OME-Zarr files into one.
    The different OME-Zarr files are expected to contain different channels of the same data set and as such have the
    same shape ([1, 1, <depth>, <height>, <width>]), the same number of resolutions and follow the following naming
    scheme:
        <out_path>_channel_<channel_number>

    The given target shape should be the same as the shape of the individual OME-Zarr files except for the number of
    channels:
        [1, <num_channel>, <depth>, <height>, <width>]

    By default, the individual OME-Zarr files are removed from the file system after applying this operation.

    @param out_path: the path of the combined OME-Zarr, and the base path of the individual OME-Zarr files (<out_path>_channel_<channel_number>)
    @param num_scales: the number of resolution levels in each of the OME-Zarr files
    @param target_shape: the target shape of the resulting OME-Zarr file.
    @param remove_individual_files: indicates if the individual OME-Zarr files should be removed from the file system after combinding them. Defaults to True.
    """
    if remove_individual_files:
        #os.replace(f'{out_path}_channel_0', f'{out_path}')
        shutil.copytree(f'{out_path}_channel_0', f'{out_path}')
    else:
        shutil.copytree(f'{out_path}_channel_0', f'{out_path}')

    for s in range(num_scales):
        with open(f'{out_path}/{s}/.zarray', 'r') as zarray_file:
            meta = json.load(zarray_file)

        meta['shape'][1] = target_shape[1]
        with open(f'{out_path}/{s}/.zarray', 'w') as zarray_file:
            json.dump(meta, zarray_file, sort_keys=True, indent=4)

    for c in range(1, target_shape[1]):
        print(c, num_scales)
        for s in range(num_scales):
            print('copying channel', c, f'{out_path}_channel_{c}/{s}/0/0', f'{out_path}/{s}/0/{c}')
            shutil.move(f'{out_path}_channel_{c}/{s}/0/0', f'{out_path}/{s}/0/{c}')
        if remove_individual_files:
            pass
            #shutil.rmtree(f'{out_path}_channel_{c}')


class DataSource(ABC):
    @abstractmethod
    def get_channel(self, channel_index: int) -> da.Array:
        pass

    @abstractmethod
    def get_shape_5d(self) -> ShapeLike5d:
        pass


class ImarisDataSource(DataSource):
    def __init__(self, file_path: FilePath, resolution_level_lock=0):
        self._ims_file = ims(file_path, ResolutionLevelLock=resolution_level_lock)
        self._resolution_level_lock = resolution_level_lock

    def get_channel(self, channel_index: int) -> da.Array:
        return da.from_array(
            np.array(self._ims_file.hf
                ['DataSet']
                [f'ResolutionLevel {self._resolution_level_lock}']
                ['TimePoint 0']
                [f'Channel {channel_index}']
                ['Data']
            )
        ).rechunk([1, 1] + list(_DASK_CHUNK_SHAPE))

    def get_shape_5d(self) -> ShapeLike5d:
        return self._ims_file.shape


class OmeTiffDataSource(DataSource):
    def __init__(self, file_path: FilePath):
        # tiff is expected to have 4 dimensions (channel, depth, height, width)
        with tf.TiffFile(file_path) as tiff_file:
            print(tiff_file, tiff_file.series[0].axes.upper(), 'CZYX'[:4-len(tiff_file.series[0].axes)] + tiff_file.series[0].axes.upper())
            self._data = move_axes(
                # todo: add reshape parameter
                da.reshape(dask_image.imread.imread(file_path), (1402, 3, 5192, 2947,)),
                'CZYX'[:4-len(tiff_file.series[0].axes)] + tiff_file.series[0].axes.upper()
            ).rechunk([1] + list(_DASK_CHUNK_SHAPE))
        print('read ome-tiff', self._data)

        #with tf.TiffFile(file_path) as tiff_file:
        #    foo = da.from_array(tf.imread(file_path))
        #    gc.collect()
        #    print('read ome-tiff')
        #    self._data = move_axes(foo, 'CZYX'[:4-len(tiff_file.series[0].axes)] + tiff_file.series[0].axes.upper())
        #    print('self._data initialized', type(self._data))

    def get_channel(self, channel_index: int) -> da.Array:
        return self._data[channel_index]

    def get_shape_5d(self) -> ShapeLike5d:
        return [1] + list(self._data.shape)


class RawDataSource(DataSource):
    def __init__(self, files: [FilePath], shape: ShapeLike3d, dtype: DTypeLike, axis_order='ZYX', num_duplicates=1):
        assert len(axis_order) == 3, f'Expected axis order to have length 3, got {axis_order}'
        assert all([a in axis_order.upper() for a in'ZYX']), f'Expected axis order to contain X, Y, and Z, got {axis_order.upper()}'
        self._shape_5d = [1, num_duplicates * len(files)] + list(shape)
        self._dtype = dtype
        self._axis_order = axis_order
        self._num_duplicates = num_duplicates
        self._file_paths = files

    def get_channel(self, channel_index: int) -> da.Array:
        return move_axes(read_raw(self._file_paths[floor(channel_index / self._num_duplicates)], self._shape_5d[2:], self._dtype), f'TC{self._axis_order}')

    def get_shape_5d(self) -> ShapeLike5d:
        return self._shape_5d


def convert_to_ome_zarr(data_source: DataSource, out_path: FilePath,
                        write_channels_as_separate_files=True,
                        chunk_shape: ShapeLike3d = None,
                        coordinate_transformations: List[List[Dict[str, Any]]] = None,
                        interpolator=sitk.sitkLinear,
                        verbose=False):
    if chunk_shape is None:
        chunk_shape = [32, 32, 32]

    shape_5d = data_source.get_shape_5d()
    chunk_shape_5d = [1, 1] + list(chunk_shape)

    multiscale = compute_multiscale_3d(shape_5d[2:], chunk_shape)
    if verbose:
        print('computed multiscale:', multiscale)

    scale_factors_between_levels = []
    for i, scale in enumerate(multiscale[1:]):
        scale_factors_between_levels.append([multiscale[i][s] / scale[s] for s in range(len(scale))])
    if verbose:
        print('scale factors between levels:', scale_factors_between_levels)

    if not write_channels_as_separate_files:
        pyramid = []

    for i in range(shape_5d[1]):
        if verbose:
            print('processing channel', i)
        if write_channels_as_separate_files:
            # free any memory from previous iterations
            pyramid = []
            gc.collect()

        # todo: remove this!!!
        if i == 0:
            continue

        if i != 0 and isinstance(data_source, RawDataSource) and write_channels_as_separate_files:
            shutil.copytree(f'{out_path}_channel_0', f'{out_path}_channel_{i}')
            continue

        volume = data_source.get_channel(i)
        if verbose:
            print('read channel', i)

        if not write_channels_as_separate_files:
            volume = volume.reshape([1, 1] + list(volume.shape))
            if verbose:
                print('... and reshaped it to', volume.shape)

        print('debug', volume)

        #volume_dtype = volume.dtype
        volume_shape = volume.shape
        #print(volume_dtype, volume_shape)
        #tmp_filename = os.path.join(mkdtemp(), '__tmp_volume__.raw')
        #volume.tofile(tmp_filename)
        #del volume
        #gc.collect()

        #volume = np.memmap(tmp_filename, dtype=volume_dtype, shape=volume_shape, mode='r')

        if i == 0 or write_channels_as_separate_files:
            pyramid.append(volume)
        else:
            pyramid[0] = np.append(pyramid[0], volume, axis=1)

        # free memory as early as possible
        del volume
        gc.collect()

        for j, s in enumerate(multiscale[1:]):
            #if j < 4:
            #    continue
            #else:
            #    continue
            if verbose:
                print('computing resolution level', j + 1)
            scale = [1. / dim for dim in s]
            zoom = scale_factors_between_levels[j]  # scale if write_channels_as_separate_files else [1., 1.] + scale

            print('debug', scale, zoom)
            resampled_shape = (
            int(volume_shape[0] * scale[0]), int(volume_shape[1] * scale[1]), int(volume_shape[2] * scale[2]),)
            print('debug', resampled_shape)

            resampled = da.from_array(ndimage.zoom(
                pyramid[j],
                zoom=zoom,
                order=1,  # default = 3
                prefilter=False  # prefiltering would produce better quality but allocates 64-bit float arrays
            )).rechunk(_DASK_CHUNK_SHAPE if write_channels_as_separate_files else [1, 1] + list(_DASK_CHUNK_SHAPE))

            if verbose:
                print('resampled volume')

            print('debug', resampled_shape, resampled)

            #resampled.tofile(f'{out_path}_channel_{i}_resampled_{j}')

            if i == 0 or write_channels_as_separate_files:
                pyramid.append(resampled)
            else:
                level = j + 1
                pyramid[level] = np.append(pyramid[level], resampled, axis=1)

            gc.collect()

        gc.collect()

        if write_channels_as_separate_files:
            if verbose:
                print('writing channel', i)

            with dask.config.set(**{'array.slicing.split_large_chunks': False}):
                for level_index, pyramid_level in enumerate(pyramid):
                    pyramid[level_index] = pyramid[level_index].reshape([1, 1] + list(pyramid_level.shape))
            gc.collect()

            write_ome_zarr(f'{out_path}_channel_{i}',
                           pyramid,
                           chunk_shape_5d,
                           coordinate_transformations=coordinate_transformations)

        if verbose:
            print('finished processing channel', i)

        gc.collect()

    print(shape_5d)
    if write_channels_as_separate_files:
        combine_ome_zarr_channels(out_path, len(multiscale), shape_5d)
    else:
        write_ome_zarr(out_path, pyramid, chunk_shape_5d, coordinate_transformations=coordinate_transformations)
