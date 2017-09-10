import os
import string
import random
import unittest
from datetime import date, timedelta
import urllib2

import gdal
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.filters import gaussian
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


def fr_data_enlarge():
    sites = ["bkl", "grz", "hbg", "ktl"]
    fr_dem_fn = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/fr_mask_dem.tif"
    fr_veg_fn = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/fr_mask_veg.tif"
    for site in sites:
        old_dir = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/fr_data/{0}".format(site)
        new_dir = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/fr_data/{0}_new".format(site)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        # dem

        print "{0} DEM".format(site)
        old_fn = os.path.join(old_dir, "dem.tif")
        new_fn = os.path.join(new_dir, "dem.tif")
        old_ds = gdal.Open(old_fn)
        old_gt = old_ds.GetGeoTransform()
        old_shp = old_ds.ReadAsArray().shape
        ulx = old_gt[0]
        uly = old_gt[3]
        lrx = old_gt[0] + old_gt[1] * (old_shp[1] + int(0.2 * old_shp[1]))
        lry = old_gt[3] + old_gt[5] * (old_shp[0] + int(0.2 * old_shp[0]))
        RU.clip_raster(fr_dem_fn, new_fn, ulx, uly, lrx, lry)

        # slope
        print "{0} SLOPE".format(site)
        new_slope_fn = os.path.join(new_dir, "slope.tif")
        RU.dem_to_slope(new_fn, new_slope_fn, scale=108000)

        # aspect
        print "{0} ASPECT".format(site)
        new_aspect_fn = os.path.join(new_dir, "aspect.tif")
        RU.dem_to_aspect(new_fn, new_aspect_fn)

        # vegetation
        print "{0} VEG".format(site)
        new_canopy_fn = os.path.join(new_dir, "canopy.tif")
        RU.clip_raster(fr_veg_fn, new_canopy_fn, ulx, uly, lrx, lry)


def resample_tuolumne_dem_test():
    RU.resample_raster_by_resolution("/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/Tuolumne_2014_bareDEM_3p0m_agg_EXPORT.tif",
                                     "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/Tuolumne_2014_bareDEM_0p001deg_agg_EXPORT.tif",
                                     x_res=0.001, y_res=0.001, dst_epsg=4326)


def delineate_tuolumne_dem():
    RU.delineate_raster_to_polygon("/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/Tuolumne_2014_bareDEM_0p001deg_agg_EXPORT.tif",
                                   "/Users/zeshizheng/Google Drive/dev/im-magic/data/tr_watershed/tr_watershed.shp",
                                   name="tuolumne_river")


def francesco_processing():

    def get_features(df, dict_arr, gt):
        x = df["point_x"].as_matrix()
        y = df["point_y"].as_matrix()
        y_idx = np.floor((y - gt[3]) / gt[5]).astype(int)
        x_idx = np.floor((x - gt[0]) / gt[1]).astype(int)
        for key, arr in dict_arr.iteritems():
            df[key] = arr[y_idx, x_idx]
        return df

    lower_met = pd.read_csv("/Users/zeshizheng/Google Drive/dev/im-magic/data/francesco/LowerMetNodes.csv")
    upper_met = pd.read_csv("/Users/zeshizheng/Google Drive/dev/im-magic/data/francesco/UpperMetNodes.csv")
    wsn_s, wsn_r, fields = RU.load_shapefile("/Users/zeshizheng/Google Drive/dev/im-magic/data/francesco/wireless_sensor_network/WSN.shp",
                                     fields=True)
    print fields
    x = []
    y = []
    field_1 = []
    field_2 = []
    field_3 = []
    for s, r in zip(wsn_s, wsn_r):
        x.append(s.points[0][0])
        y.append(s.points[0][1])
        field_1.append(r[0])
        field_2.append(r[1])
        field_3.append(r[2])
    wsn_df = pd.DataFrame({"point_x":x,
                           "point_y":y,
                           fields[1][0]: field_1,
                           fields[2][0]: field_2,
                           fields[3][0]: field_3})
    hh = gdal.Open("/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/rasters_sdem/output_hh.tif")
    dem = gdal.Open("/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/rasters_sdem/output_be.tif")
    slp = gdal.Open("/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/rasters_sdem/output_slp.tif")
    asp = gdal.Open("/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/rasters_sdem/output_asp.tif")
    dem_arr = dem.ReadAsArray()
    hh_arr = hh.ReadAsArray()
    slp_arr = slp.ReadAsArray()
    slp_arr[slp_arr == slp.GetRasterBand(1).GetNoDataValue()] = np.nan
    asp_arr = asp.ReadAsArray()
    asp_arr[asp_arr == asp.GetRasterBand(1).GetNoDataValue()] = np.nan
    plt.imshow(asp_arr)
    plt.show()
    dem_arr[dem_arr == dem.GetRasterBand(1).GetNoDataValue()] = np.nan
    hh_arr[hh_arr == hh.GetRasterBand(1).GetNoDataValue()] = np.nan
    chm_arr = hh_arr - dem_arr
    chm_arr[chm_arr < 0.] = 0.
    chm_arr[chm_arr > 150.] = 0.
    blurred_binary_arr = gaussian((chm_arr >= 5.0).astype(float), sigma=5)
    dem_gt = dem.GetGeoTransform()
    features_dict = {"dem": dem_arr, "slp": slp_arr, "asp": asp_arr, "veg": blurred_binary_arr}
    lower_met = get_features(lower_met, features_dict, dem_gt)
    upper_met = get_features(upper_met, features_dict, dem_gt)
    wsn_df = get_features(wsn_df, features_dict, dem_gt)
    lower_met.to_csv("/Users/zeshizheng/Google Drive/dev/im-magic/data/francesco/lower_met_features.csv")
    upper_met.to_csv("/Users/zeshizheng/Google Drive/dev/im-magic/data/francesco/upper_met_features.csv")
    wsn_df.to_csv("/Users/zeshizheng/Google Drive/dev/im-magic/data/francesco/wsn_features.csv")

francesco_processing()
