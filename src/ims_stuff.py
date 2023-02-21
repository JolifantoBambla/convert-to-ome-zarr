# this seemingly unused import is required for reading ims file data without crashing (yay Python ecosystem <3)
import hdf5plugin
import imaris_ims_file_reader.ims as ims
import numpy as np

# the file has 6 resolution levels -> use the lowest res for playing around
ims_file = ims('imaris.ims', ResolutionLevelLock=5)

# imaris files are stored as hdf5 files
list(ims_file.hf['DataSet']['ResolutionLevel 5']['TimePoint 0']['Channel 0']['Histogram'])
# >>> [292757, 39913, 26291, 18117, 11761, 6932, 3711, 1709, 753, 302, 129, 57, 24, 15, 5, 0, 2, ...]

np.any(np.array(ims_file.hf['DataSet']['ResolutionLevel 5']['TimePoint 0']['Channel 0']['Data']) == 8)
# >>> True

np.any(np.array(ims_file.hf['DataSet']['ResolutionLevel 5']['TimePoint 0']['Channel 0']['Data']) == 15)
# >>> False

ims_file_2 = ims('/home/lherzberger/Data/ims/.ims', ResolutionLevelLock=0)
channel_1 = np.array(ims_file.hf['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 0']['Data'])
