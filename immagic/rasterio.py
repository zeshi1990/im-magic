import os
import numbers
import string
import random
import copy

import gdal
import osr
import numpy as np
from gdal import GA_ReadOnly

from raster_utils import RasterUtils as RU

os.environ['GDAL_DATA'] = os.popen('gdal-config --datadir').read().rstrip()
rand_str = lambda n: ''.join([random.choice(string.lowercase) for i in xrange(n)])


class RasterDataSetUtils(object):

    @classmethod
    def read_raster(cls, fn):
        ds = gdal.Open(fn, GA_ReadOnly)
        return RasterDataSet(ds)

    @classmethod
    def reproject_raster_epsg(cls, src, dst_epsg):
        """

        :param src:
        :param dst_epsg:
        :return:
        """
        raise NotImplementedError

    @classmethod
    def reproject_raster(cls, src_ds, match_ds, gra_type=gdal.GRA_Bilinear):
        """
         Reproject a raster with reference of a matching raster data

         Parameters
         ----------
         src_ds : RasterDataSet, source filename to reproject from
         match_ds : RasterDataSet, match filename that the reprojected file's projection and geotransform will
                    match with it
         gra_type : int, gdal.GRA_***

         Returns
         -------
         dst_ds_rd: RasterDataSet

        """
        # Source

        src_proj = src_ds.ds.GetProjection()
        src_n_bands = src_ds.ds.RasterCount

        # We want a section of source that matches this:

        match_proj = match_ds.ds.GetProjection()
        match_geotrans = match_ds.ds.GetGeoTransform()
        wide = match_ds.ds.RasterXSize
        high = match_ds.ds.RasterYSize

        # Output / destination
        dst_ds = gdal.GetDriverByName("MEM").Create("", wide, high, src_n_bands, GDT_Float32)
        dst_ds.SetGeoTransform(match_geotrans)
        dst_ds.SetProjection(match_proj)
        for b in range(src_n_bands):
            dst_ds.GetRasterBand(b + 1).SetNoDataValue(-9999.)

        # Do the work
        gdal.ReprojectImage(src_ds.ds, dst_ds, src_proj, match_proj, gra_type)
        dst_ds_rd = RasterDataSet(dst_ds)
        return dst_ds_rd

    @classmethod
    def mask_raster(cls, src, mask, dst_epsg):
        """

        :param src:
        :param mask:
        :param dst_epsg:
        :return:
        """
        raise NotImplementedError

    @classmethod
    def mosaic_rasters(cls, l_rasters, srcNodata=-9999., dstNodata=-9999., srcSRS=4326, dstSRS=4326):
        """
        Merge a few rasters together
        """
        # make some initial assertions so that the function will run smoothly
        assert isinstance(l_rasters, list), "l_rasters has to be of type list, isinstance(l_raster, list) = False " \
                                            "and type(l_raster) = {0} != list".format(type(l_rasters))
        assert len(l_rasters) >= 2, "l_rasters has to have length larger or equal to 2, len(l_rasters) = " \
                                    "{0} < 2".format(len(l_rasters))

        for i, item in enumerate(l_rasters):
            assert isinstance(item, RasterDataSet), "The item in l_rasters has to be a RasterDataSet instance, " \
                                                    "type(#{0}item) = {1} != RasterDataSet".format(i, type(item))
            assert item.epsg == srcSRS, "The item in l_rasters does not have the specified Projection"

        warpopts = gdal.WarpOptions(srcNodata=srcNodata, dstNodata=dstNodata,
                                    srcSRS="EPSG:{0}".format(srcSRS), dstSRS="EPSG:{0}".format(dstSRS))
        tmp_mosaic_fn = "/tmp/mosaic_{0}.tif".format(rand_str(12))
        gdal.Warp(l_rasters, tmp_mosaic_fn, options=warpopts)
        dst_ds = gdal.Open(tmp_mosaic_fn)
        mosaiced_ds = RasterDataSet(dst_ds)
        os.remove(tmp_mosaic_fn)
        return mosaiced_ds


class RasterDataSet(object):

    def __init__(self, ds):
        """
        Initialize the raster data set

        Parameters
        ----------
        ds : gdal.Dataset, a gdal data set

        """
        assert isinstance(ds, gdal.Dataset), "ds is not a instance of gdal.Dataset"
        self.ds = ds
        self.geotransform = ds.GetGeoTransform()
        self.projection = ds.GetProjection()
        self.values = np.rollaxis(ds.ReadAsArray(), 0, 3)
        self.n_bands = ds.RasterCount
        self.nodata = ds.GetRasterBand(1).GetNoDataValue()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.projection)
        self.epsg = int(srs.GetAttrValue("AUTHORITY", 1))

    def save(self, path):
        RU.write_array_to_raster(path, self.values, self.projection, self.geotransform)
        return 0

    def __call__(self):
        return self.values

    def __add__(self, other):
        """
         to make an add algebra of rasterdata

        :param other:
        self:rasterdataset
        other:rasterdataset of number

        :return: the summation of two rasterdataset or a rasterdataset and a number

        """
        assert isinstance(other, RasterDataSet) or isinstance(other, numbers.Number), \
            "other has to be a RasterDataSet or a numbers.Number type"
        if isinstance(other, numbers.Number):
            self.values[self.values != -9999] += other
            return self
        else:
            trans_rs_rd = RasterDataSetUtils.reproject_raster(other, self)
            dst = self
            dst.values[np.logical_and(dst.values != -9999, trans_rs_rd.values != -9999)] += \
                trans_rs_rd.values[np.logical_and(dst.values != -9999, trans_rs_rd.values != -9999)]
            return dst

    def __sub__(self, other):
        """
         to make a sub algebra of rasterdata

        :param other:
        self:rasterdataset
        other:rasterdataset of number

        :return: the summation of two rasterdataset or a rasterdataset and a number

        """
        assert isinstance(other, RasterDataSet) or isinstance(other, numbers.Number), \
            "other has to be a RasterDataSet or a numbers.Number type"
        if isinstance(other, numbers.Number):
            self.values[self.values != -9999] -= other
            return self
        else:
            trans_rs_rd = RasterDataSetUtils.reproject_raster(other, self)
            dst = self
            dst.values[np.logical_and(dst.values != -9999, trans_rs_rd.values != -9999)] -= \
                trans_rs_rd.values[np.logical_and(dst.values != -9999, trans_rs_rd.values != -9999)]
            return dst

    def __mul__(self, other):
        """
         to make a mul algebra of rasterdata

        :param other:
        self:rasterdataset
        other:rasterdataset of number

        :return: the summation of two rasterdataset or a rasterdataset and a number

        """
        assert isinstance(other, RasterDataSet) or isinstance(other, numbers.Number), \
            "other has to be a RasterDataSet or a numbers.Number type"
        if isinstance(other, numbers.Number):
            self.values[self.values != -9999] *= other
            return self
        else:
            trans_rs_rd = RasterDataSetUtils.reproject_raster(other, self)
            dst = self
            dst.values[np.logical_and(dst.values != -9999, trans_rs_rd.values != -9999)] *= \
                trans_rs_rd.values[np.logical_and(dst.values != -9999, trans_rs_rd.values != -9999)]
            return dst

    def __div__(self, other):
        """
         to make a div algebra of rasterdata

        :param other:
        self:rasterdataset
        other:rasterdataset of number

        :return: the summation of two rasterdataset or a rasterdataset and a number

        """
        assert isinstance(other, RasterDataSet) or isinstance(other, numbers.Number), \
            "other has to be a RasterDataSet or a numbers.Number type"
        if isinstance(other, numbers.Number):
            self.values[self.values != -9999] /= other
            return self
        else:
            trans_rs_rd = RasterDataSetUtils.reproject_raster(other, self)
            dst = self
            dst.values[np.logical_and(dst.values != -9999, trans_rs_rd.values != -9999)] /= \
                trans_rs_rd.values[np.logical_and(dst.values != -9999, trans_rs_rd.values != -9999)]
            return dst

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)
