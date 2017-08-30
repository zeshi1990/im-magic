import os
import string
import random
import unittest
from datetime import date, timedelta

import gdal
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import canny
from skimage.measure import find_contours

from immagic import RasterUtils as RU
from immagic import RasterDataSet as RDS
from immagic.rasterio import RasterDataSetUtils as RDSU

os.environ['GDAL_DATA'] = os.popen('gdal-config --datadir').read().rstrip()
#
# print RU.get_shapetypes()
# print type(RU.get_epsg("/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/amr_forest_density_30m.tif"))


def delineate_raster_to_polygon_test():
    RU.delineate_raster_to_polygon("/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/amr_watershed_clipped.tif",
                                   "/Users/zeshizheng/Google Drive/dev/im-magic/data/ar_watershed/amr_watershed.shp",
                                   "american_river")


def mask_hdf5_to_tif_test():
    hdf5_fn = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/SN_SWE_WY2016.h5"
    mask_shp = "/Users/zeshizheng/Google Drive/dev/im-magic/data/ar_watershed/amr_watershed.shp"
    match_raster = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/amr_watershed_clipped.tif"
    dst_fn = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/amr_sn_swe/amr_sn_swe_{0}.tif"
    dst_fns = []
    start_date = date(2015, 10, 1)
    for i in range(365):
        temp_date = start_date + timedelta(days=i)
        temp_date_str = temp_date.strftime("%Y%m%d")
        dst_fns.append(dst_fn.format(temp_date_str))
    RU.mask_hdf5(hdf5_fn, mask_shp, match_raster, dst_fns, hdf5_idx=range(365))







# match_raster = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/amr_watershed_clipped.tif"
# print gdal.Open(match_raster).GetProjection()
#
# warpopts = gdal.WarpOptions(dstSRS='EPSG:{0}'.format(4326))
# gdal.Warp("/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/amr_watershed_clipped_reprojected.tif",
#           "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/amr_watershed_clipped.tif",
#           options=warpopts)
# print gdal.Open("/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/amr_watershed_clipped_reprojected.tif").GetProjection()

# ds = gdal.Open("/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/ar_sn_swe_20160101.tif")
# print ds.GetProjection()
# print ds.GetGeoTransform()
# print ds.ReadAsArray()
# plt.imshow(ds.ReadAsArray())
# plt.colorbar()
# plt.show()
# print np.unique(ds.ReadAsArray())


# dem_arr = ar_dem.ReadAsArray()
# plt.imshow(dem_arr)
# plt.show()
# print dem_arr

# new_dem_arr = dem_arr[100:1000, 200:]
# new_gt = RU.calculate_new_geotransform(ar_dem.GetGeoTransform(), 100, 200)
# new_dem_arr[new_dem_arr < -3e38] = -9999.
# RU.write_array_to_raster("/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/amr_watershed_clipped.tif",
#                          new_dem_arr, ar_dem.GetProjection(), new_gt)

