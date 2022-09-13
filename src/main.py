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
from ome_zarr.writer import write_image

@dataclass
class RawFileInfo:
    path: Union[str, bytes, PathLike]
    shape: Tuple[int, int, int]
    data_type: DTypeLike


def read_raw(file_name: Union[str, bytes, PathLike], shape: Tuple[int, int, int], dtype: DTypeLike) -> NDArray:
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


def write_as_ome_tiff(out_path: Union[str, bytes, PathLike], images: List[NDArray], tile_shape: Tuple[int, int, int]):
    """

    :param out_path:
    :param images:
    :param tile_shape: (depth, height, width)
    """
    base_shape = np.shape(images[0])
    data_type = images[0].dtype

    kwargs = {}

    for i, img in enumerate(images):
        img_shape = np.shape(img)
        img_dtype = img.dtype

        if img_shape != base_shape:
            raise RuntimeError(f"Expected shape {base_shape} to match first input image, got {img_shape} instead.")

        if img_dtype != data_type:
            raise RuntimeError(f"Expected dtype '{data_type}' to match first input image, got '{img_dtype}' instead.")

        tf.imwrite(
            out_path, img, bigtiff=True, append=True, tile=tile_shape,
            ome=True, imagej=False, metadata=None, **kwargs
        )


def write_as_ome_zarr(out_path: Union[str, bytes, PathLike], images: List[NDArray], tile_shape: Tuple[int, int, int]):
    """

    :param out_path:
    :param images:
    :param tile_shape: (depth, height, width)
    """
    base_shape = np.shape(images[0])
    data_type = images[0].dtype

    kwargs = {}

    scaler = Scaler(method='zoom')
    print('downsampling')
    downsampling = scaler.zoom(images[0])
    print('downsampled')
    for x in downsampling:
        print(x.shape)

    #store = parse_url(out_path, mode="w").store
    #root = zarr.group(store=store)
    #write_image(
    #    image=images[0],
    #    group=root,
    #    axes="tczyx",
    #    scaler=Scaler(method='zoom'),
    #    chunks=tile_shape,
    #    storage_options=dict(chunks=tile_shape),
    #)
    #root.attrs["omero"] = {
    #    "channels": [{
    #        "color": "00FFFF",
    #        "window": {"start": 0, "end": 20},
    #        "label": "random",
    #        "active": True,
    #    }]
    #}


def read_volume(file_path, data_type=sitk.sitkUInt16) -> sitk.Image:
    return sitk.ReadImage(file_path, data_type)  # read and cast to dtype


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


def convert_image(input_file_name, output_file_name, new_width=None):
    try:
        image_file_reader = sitk.ImageFileReader()
        # only read DICOM images
        image_file_reader.SetImageIO("GDCMImageIO")
        image_file_reader.SetFileName(input_file_name)
        image_file_reader.ReadImageInformation()
        image_size = list(image_file_reader.GetSize())
        if len(image_size) == 3 and image_size[2] == 1:
            image_size[2] = 0
        image_file_reader.SetExtractSize(image_size)
        image = image_file_reader.Execute()
        if new_width:
            original_size = image.GetSize()
            original_spacing = image.GetSpacing()
            new_spacing = [
                (original_size[0] - 1) * original_spacing[0] / (new_width - 1)
            ] * 2
            new_size = [
                new_width,
                int(
                    (original_size[1] - 1)
                    * original_spacing[1]
                    / new_spacing[1]
                ),
            ]
            image = sitk.Resample(
                image1=image,
                size=new_size,
                transform=sitk.Transform(),
                interpolator=sitk.sitkLinear,
                outputOrigin=image.GetOrigin(),
                outputSpacing=new_spacing,
                outputDirection=image.GetDirection(),
                defaultPixelValue=0,
                outputPixelType=image.GetPixelID(),
            )
        # If a single channel image, rescale to [0,255]. Also modify the
        # intensity values based on the photometric interpretation. If
        # MONOCHROME2 (minimum should be displayed as black) we don't need to
        # do anything, if image has MONOCRHOME1 (minimum should be displayed as
        # white) we flip # the intensities. This is a constraint imposed by ITK
        # which always assumes MONOCHROME2.
        if image.GetNumberOfComponentsPerPixel() == 1:
            image = sitk.RescaleIntensity(image, 0, 255)
            if (
                image_file_reader.GetMetaData("0028|0004").strip()
                == "MONOCHROME1"
            ):
                image = sitk.InvertIntensity(image, maximum=255)
            image = sitk.Cast(image, sitk.sitkUInt8)
        sitk.WriteImage(image, output_file_name)
        return True
    except BaseException:
        return False


def convert_images(input_file_names, output_file_names, new_width):
    MAX_PROCESSES = 15
    with multiprocessing.Pool(processes=MAX_PROCESSES) as pool:
        return pool.starmap(
            functools.partial(convert_image, new_width=new_width),
            zip(input_file_names, output_file_names),
        )


def positive_int(int_str):
    value = int(int_str)
    if value <= 0:
        raise argparse.ArgumentTypeError(
            int_str + " is not a positive integer value"
        )
    return value


def directory(dir_name):
    if not os.path.isdir(dir_name):
        raise argparse.ArgumentTypeError(
            dir_name + " is not a valid directory name"
        )
    return dir_name


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert and resize DICOM files to common image types."
    )
    parser.add_argument(
        "root_of_data_directory",
        type=directory,
        help="Path to the topmost directory containing data.",
    )
    parser.add_argument(
        "output_file_extension",
        help="Image file extension, this determines output file type "
        "(e.g. png) .",
    )
    parser.add_argument(
        "--w", type=positive_int, help="Width of converted images."
    )
    parser.add_argument("--od", type=directory, help="Output directory.")
    args = parser.parse_args(argv)

    input_file_names = []
    for dir_name, subdir_names, file_names in os.walk(
        args.root_of_data_directory
    ):
        input_file_names += [
            os.path.join(os.path.abspath(dir_name), fname)
            for fname in file_names
        ]
    if args.od:
        # if all output files are written to the same directory we need them
        # to have a unique name, so use an index.
        file_names = [
            os.path.join(os.path.abspath(args.od), str(i))
            for i in range(len(input_file_names))
        ]
    else:
        file_names = input_file_names
    output_file_names = [
        file_name + "." + args.output_file_extension
        for file_name in file_names
    ]

    res = convert_images(input_file_names, output_file_names, args.w)
    input_file_names = list(itertools.compress(input_file_names, res))
    output_file_names = list(itertools.compress(output_file_names, res))

    # save csv file mapping input and output file names.
    # using csv module and not pandas so as not to create more dependencies
    # for the examples. pandas based code is more elegant/shorter.
    dir_name = args.od if args.od else os.getcwd()
    with open(os.path.join(dir_name, "file_names.csv"), mode="w") as fp:
        fp_writer = csv.writer(
            fp, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        fp_writer.writerow(["input file name", "output file name"])
        for data in zip(input_file_names, output_file_names):
            fp_writer.writerow(data)


# https://simpleitk.readthedocs.io/en/master/link_RawImageReading_docs.html
def read_raw(
    binary_file_name,
    image_size,
    sitk_pixel_type,
    image_spacing=None,
    image_origin=None,
    big_endian=False,
):
    """
    Read a raw binary scalar image.

    Parameters
    ----------
    binary_file_name (str): Raw, binary image file content.
    image_size (tuple like): Size of image (e.g. [2048,2048])
    sitk_pixel_type (SimpleITK pixel type: Pixel type of data (e.g.
        sitk.sitkUInt16).
    image_spacing (tuple like): Optional image spacing, if none given assumed
        to be [1]*dim.
    image_origin (tuple like): Optional image origin, if none given assumed to
        be [0]*dim.
    big_endian (bool): Optional byte order indicator, if True big endian, else
        little endian.

    Returns
    -------
    SimpleITK image or None if fails.
    """

    pixel_dict = {
        sitk.sitkUInt8: "MET_UCHAR",
        sitk.sitkInt8: "MET_CHAR",
        sitk.sitkUInt16: "MET_USHORT",
        sitk.sitkInt16: "MET_SHORT",
        sitk.sitkUInt32: "MET_UINT",
        sitk.sitkInt32: "MET_INT",
        sitk.sitkUInt64: "MET_ULONG_LONG",
        sitk.sitkInt64: "MET_LONG_LONG",
        sitk.sitkFloat32: "MET_FLOAT",
        sitk.sitkFloat64: "MET_DOUBLE",
    }
    direction_cosine = [
        "1 0 0 1",
        "1 0 0 0 1 0 0 0 1",
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
    ]
    dim = len(image_size)
    header = [
        "ObjectType = Image\n".encode(),
        (f"NDims = {dim}\n").encode(),
        (
            "DimSize = " + " ".join([str(v) for v in image_size]) + "\n"
        ).encode(),
        (
            "ElementSpacing = "
            + (
                " ".join([str(v) for v in image_spacing])
                if image_spacing
                else " ".join(["1"] * dim)
            )
            + "\n"
        ).encode(),
        (
            "Offset = "
            + (
                " ".join([str(v) for v in image_origin])
                if image_origin
                else " ".join(["0"] * dim) + "\n"
            )
        ).encode(),
        ("TransformMatrix = " + direction_cosine[dim - 2] + "\n").encode(),
        ("ElementType = " + pixel_dict[sitk_pixel_type] + "\n").encode(),
        "BinaryData = True\n".encode(),
        ("BinaryDataByteOrderMSB = " + str(big_endian) + "\n").encode(),
        # ElementDataFile must be the last entry in the header
        (
            "ElementDataFile = " + os.path.abspath(binary_file_name) + "\n"
        ).encode(),
    ]
    fp = tempfile.NamedTemporaryFile(suffix=".mhd", delete=False)

    print(header)

    # Not using the tempfile with a context manager and auto-delete
    # because on windows we can't open the file a second time for ReadImage.
    fp.writelines(header)
    fp.close()
    img = sitk.ReadImage(fp.name)
    os.remove(fp.name)
    return img


if __name__ == '__main__':
    # For raw image (without any headers)
    in_path = "/home/ripley/Documents/data/raw/bunny_512x512x361_uint16.raw"
    dtype = np.uint16
    shape = 1, 1, 361, 512, 512,

    out_path = "/tmp/bunny.zarr"
    tile_shape = 1, 1, 32, 128, 128,

    #write_as_ome_zarr(
    #    out_path,
    #    read_files([RawFileInfo(in_path, shape, dtype)]),
    #    tile_shape
    #)

    input_shape = 316, 512, 512,
    current_shape = input_shape
    target_shape = 32, 32, 32,
    spacings = [[1., 1., 1.]]

    while any([current_shape[i] > target_shape[i] for i in range(3)]):
        last_spacing = spacings[-1:][0]
        spacing = [s * 2. if all([current_shape[i] >= current_shape[j] / 2. for j in range(3) if i != j]) else 1. for i, s in enumerate(last_spacing)]
        spacings.append(spacing)
        current_shape = [n / s for n, s in zip(input_shape, spacing)]

    volume = read_raw(in_path, list(shape[2:]), sitk.sitkUInt16)
    print(np.shape(sitk_to_numpy(volume)))

    for s in spacings:
        resampled = resample_volume(volume, new_spacing=s)
        print(np.shape(sitk_to_numpy(resampled)))



    #sys.exit(main())
