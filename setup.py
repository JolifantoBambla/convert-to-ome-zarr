from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = [i.strip() for i in f.readlines()]

setup(
   name='convert_to_ome_zarr',
   version='1.0',
   description='Utility to create OME-Zarr data sets from other file formats.',
   license="MIT",
   long_description=long_description,
   author='Lukas Herzberger',
   author_email='herzberger.lukas@gmail.com',
   url="https://github.com/JolifantoBambla/raw-to-ome-zarr",
   packages=['convert_to_ome_zarr'],
   install_requires=requirements,
   scripts=[
       'scripts/ome_zarr_from_imaris',
       'scripts/ome_zarr_from_ome_tiff',
       'scripts/ome_zarr_from_raw',
   ]
)
