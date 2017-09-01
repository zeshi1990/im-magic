from matplotlib import pyplot as plt

from immagic import RasterUtils as RU

hdf5_fn = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/SN_SWE_WY2016.h5"
array, gt = RU.load_hdf5_slices(hdf5_fn, [0, 1, 2, 3])
plt.imshow(array[0])
plt.colorbar()
plt.show()
