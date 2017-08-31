import os
import string
import random
import unittest
from datetime import date, timedelta
import urllib2

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
    for i in range(364, 366):
        temp_date = start_date + timedelta(days=i)
        temp_date_str = temp_date.strftime("%Y%m%d")
        dst_fns.append(dst_fn.format(temp_date_str))
    RU.mask_hdf5(hdf5_fn, mask_shp, match_raster, dst_fns, hdf5_idx=range(364, 366))


def resample_data_test():
    test_fn = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/fr_mask_dem.tif"
    test_data = gdal.Open(test_fn)
    nodata = test_data.GetRasterBand(1).GetNoDataValue()
    dst_fn = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/fr_mask_dem_resampled.tif"
    RU.resample_raster_by_resolution(test_fn, dst_fn, 0.001, 0.001,
                                     dst_epsg=4326, src_nodata=nodata, dst_nodata=-9999.)


def delineate_feather_test():
    if not os.path.exists("/Users/zeshizheng/Google Drive/dev/im-magic/data/fr_watershed"):
        os.mkdir("/Users/zeshizheng/Google Drive/dev/im-magic/data/fr_watershed")
    RU.delineate_raster_to_polygon("/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/fr_mask_dem_resampled.tif",
                                   "/Users/zeshizheng/Google Drive/dev/im-magic/data/fr_watershed/fr_watershed.shp",
                                   "feather_river")


def paramiko_test():
    from paramiko import Transport, SFTPClient, RSAKey
    host = "glaser.berkeley.edu"
    port = 5441
    transport = Transport((host, port))
    pkey = RSAKey.from_private_key_file("/Users/zeshizheng/.ssh/id_rsa")
    transport.connect(username="zeshi", pkey=pkey)
    sftp = SFTPClient.from_transport(transport)
    sftp.get("/media/raid0/zeshi/eric.zip", "/Users/zeshizheng/Google Drive/dev/im-magic/data/eric.zip")
    sftp.close()
    transport.close()


def epsg_test():
    epsg_wkt = RU.get_epsg(4326)
    data = urllib2.urlopen("http://spatialreference.org/ref/epsg/wgs-84/prettywkt/")
    wkt = ""
    for line in data.readlines():
        wkt += line[:-1].strip()
    print wkt
    print epsg_wkt
