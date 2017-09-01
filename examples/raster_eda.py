import gdal
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat

from immagic import RasterUtils as RU

# fn = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/Tuolumne_2014_bareDEM_3p0m_agg_EXPORT.tif"
# RU.reset_raster_nodata(fn, 0.)


fn = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/Tuolumne_2014_bareDEM_0p001deg_agg_EXPORT.tif"
ds = gdal.Open(fn)
print ds.GetGeoTransform()
print ds.GetProjection()
