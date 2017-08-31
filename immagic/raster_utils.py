import os
import logging
import random
import string

import gdal
import osr
import ogr
import numpy as np
import xarray
import shapefile
from shapely.geometry import Polygon

from skimage.draw import polygon
from skimage.measure import find_contours
from gdal import GA_ReadOnly, GRA_Bilinear, GDT_Float32

os.environ['GDAL_DATA'] = os.popen('gdal-config --datadir').read().rstrip()

rand_str = lambda n: ''.join([random.choice(string.lowercase) for i in xrange(n)])

# Predefine shapefile types
SHAPETYPES = {
    0: "NULL",
    1: "POINT",
    3: "POLYLINE",
    5: "POLYGON",
    8: "MULTIPOINT",
    11: "POINTZ",
    13: "POLYLINEZ",
    15: "POLYGONZ",
    18: "MULTIPOINTZ",
    21: "POINTM",
    23: "POLYLINEM",
    25: "POLYGONM",
    28: "MULTIPOINTM",
    31: "MULTIPATCH"
}

# Initialize global osr spatial reference instance
SRS = osr.SpatialReference()

# Initialize logger
LOGGER = logging.getLogger("RasterUtils")
LOGGER.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
ch.setFormatter(formatter)
LOGGER.addHandler(ch)


class RasterUtils(object):

    @classmethod
    def load_hdf5_slices(cls, fn, hdf5_idx, hdf5_arr="SWE", hdf5_lat="lat", hdf5_lon="lon"):
        """
        Load a single slice img array from a hdf5 data set. Here we assume that the hdf5
        images are

        Parameters
        ----------

        fn : str, hdf5 file name

        hdf5_idx : int, the index of the image slice in the hdf5 data cube

        hdf5_arr : str, hdf5 image array field code

        hdf5_lat : str, hdf5 image array latitude coordinate code

        hdf5_lon : str, hdf5 image array longitude coordinate code

        Returns
        -------

        array : The slice image
        geotrans : the image's geotransform.
        """
        # Load hdf5 dataset using xarray rather than gdal
        ds = xarray.open_dataset(fn)

        # Take the slice of the image array
        if isinstance(hdf5_idx, list):
            array = ds[hdf5_arr][hdf5_idx, :, :].values.swapaxes(1, 2)
        else:
            array = ds[hdf5_arr][hdf5_idx, :, :].values.T

        # Read the coordination array
        lat = ds[hdf5_lat].values.flatten()
        lon = ds[hdf5_lon].values.flatten()

        # Encode gdal-style geotransform
        geotrans = (lon[0] - 0.5 * (lon[1] - lon[0]),
                    lon[1] - lon[0],
                    0.,
                    lat[0] - 0.5 * (lat[1] - lat[0]),
                    0.,
                    lat[1] - lat[0])
        return array, geotrans

    @classmethod
    def get_shapetypes(cls):
        """
        Get all shape types.

        Parameters
        ----------

        No params

        Returns
        -------
        The global dictionary of shape types.

        """
        LOGGER.info("Get shape types of the data")
        return SHAPETYPES

    @classmethod
    def load_hdf5(cls, fn):
        """
        Load a hdf5 xarray data set

        Parameters
        ----------

        fn : str, hdf5 data set file name.

        Returns
        -------

        xarray data set of the hdf5 data
        """
        ds = xarray.open_dataset(fn)
        return ds

    @classmethod
    def get_epsg(cls, epsg):
        """
        Get detailed projection information for a EPSG code

        Parameters
        ----------

        epsg : int

        Returns
        -------

        str, EPSG detailed information
        """
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        return srs.ExportToWkt()

    @classmethod
    def delineate_raster_to_polygon(cls, src_fn, dst_fn, name, epsg=4326):
        # Load data
        ds = gdal.Open(src_fn)
        assert isinstance(ds, gdal.Dataset)
        # Load band
        band = ds.GetRasterBand(1)
        assert isinstance(band, gdal.Band)
        gt = ds.GetGeoTransform()
        nodata = band.GetNoDataValue()

        # Image processing so we can find the delineation of the basin
        array = band.ReadAsArray()
        array_binary = array.copy()
        array_binary[array_binary != nodata] = 1.
        array_binary[array_binary == nodata] = 0.
        contour = find_contours(array_binary, level=0.5, fully_connected='high')[0]
        contour = np.vstack((contour, contour[0]))
        y_coords = gt[3] + gt[5] * contour[:, 0]
        x_coords = gt[0] + gt[1] * contour[:, 1]
        coords = []
        for x, y in zip(x_coords, y_coords):
            coords.append((x, y))

        # Generate polygon shapefile
        poly = Polygon(coords)
        driver = ogr.GetDriverByName('Esri Shapefile')
        ds = driver.CreateDataSource(dst_fn)
        dest_srs = ogr.osr.SpatialReference()
        dest_srs.ImportFromEPSG(epsg)
        layer = ds.CreateLayer('', dest_srs, ogr.wkbPolygon)

        # Add one attribute
        layer.CreateField(ogr.FieldDefn('name', ogr.OFTString))
        defn = layer.GetLayerDefn()

        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('name', name)

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkb(poly.wkb)
        feat.SetGeometry(geom)

        layer.CreateFeature(feat)

        del feat, geom, ds, layer
        return 0

    @classmethod
    def resample_raster_by_resolution(cls, src_fn, dst_fn, x_res, y_res,
                                      dst_epsg=None, src_nodata=-9999., dst_nodata=-9999.):
        warpopts = gdal.WarpOptions(xRes=x_res, yRes=y_res, srcNodata=src_nodata, dstNodata=dst_nodata,
                                    dstSRS=None if dst_epsg is None else 'EPSG:{0}'.format(dst_epsg))
        gdal.Warp(dst_fn, src_fn, options=warpopts)
        return 0

    @classmethod
    def calculate_new_geotransform(cls, src_gt, ix_y, ix_x):
        return (src_gt[0] + ix_x * src_gt[1],
                src_gt[1], src_gt[2],
                src_gt[3] + ix_y * src_gt[5],
                src_gt[4],
                src_gt[5])

    @classmethod
    def write_array_to_raster(cls, dst_fn, arr, proj, geotransform):
        """
        Write an array to a GeoTIFF file

        Parameters
        ----------

        dst_fn :

        arr :

        proj :

        geotransform :

        Returns
        -------


        """

        arr_shape = arr.shape
        num_bands = 1

        if len(arr_shape) != 2:
            num_bands = arr_shape[-1]
        else:
            arr = arr[:, :, np.newaxis]

        driver = gdal.GetDriverByName('GTiff')

        dataset = driver.Create(
            dst_fn,
            arr.shape[1],
            arr.shape[0],
            num_bands,
            gdal.GDT_Float32)

        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(proj)
        for i in range(num_bands):
            band_idx = i + 1
            dataset.GetRasterBand(band_idx).WriteArray(arr[:, :, i])
            dataset.GetRasterBand(band_idx).SetNoDataValue(-9999.)
        dataset.FlushCache()
        return 0

    @classmethod
    def reproject_raster(cls, src_fn, match_fn, dst_fn, gra_type=GRA_Bilinear):
        """
        Reproject a raster with reference of a matching raster data
        """

        # Source
        src_ds = gdal.Open(src_fn, GA_ReadOnly)
        assert isinstance(src_ds, gdal.Dataset), "{0} is not a valid gdal raster data set".format(src_fn)
        src_proj = src_ds.GetProjection()
        src_n_bands = src_ds.RasterCount

        # We want a section of source that matches this:
        match_ds = gdal.Open(match_fn, GA_ReadOnly)
        assert isinstance(match_ds, gdal.Dataset), "{0} is not a valid gdal raster data set".format(match_fn)
        match_proj = match_ds.GetProjection()
        match_geotrans = match_ds.GetGeoTransform()
        wide = match_ds.RasterXSize
        high = match_ds.RasterYSize

        # Output / destination
        dst_ds = gdal.GetDriverByName('GTiff').Create(dst_fn, wide, high, src_n_bands, GDT_Float32)
        dst_ds.SetGeoTransform(match_geotrans)
        dst_ds.SetProjection(match_proj)
        for b in range(src_n_bands):
            dst_ds.GetRasterBand(b + 1).SetNoDataValue(-9999.)

        # Do the work
        gdal.ReprojectImage(src_ds, dst_ds, src_proj, match_proj, gra_type)
        dst_ds.FlushCache()
        return 0

    @classmethod
    def clip_raster(cls, src_fn, dst_fn, ulx, uly, lrx, lry):
        opts = gdal.WarpOptions(outputBounds=[ulx, lry, lrx, uly])
        gdal.Warp(dst_fn, src_fn, options=opts)
        return 0

    @classmethod
    def reset_raster_nodata(cls, src_fn, src_nodata):
        ds = gdal.Open(src_fn, GA_ReadOnly)
        array = ds.ReadAsArray()
        proj = ds.GetProjection()
        gt = ds.GetGeoTransform()
        array[array == src_nodata] = -9999.
        cls.write_array_to_raster(src_fn, array, proj=proj, geotransform=gt)
        return 0


    @classmethod
    def dem_to_slope(cls, dem_fn, slope_fn, scale=None):
        if scale:
            opts = gdal.DEMProcessingOptions(computeEdges=True, slopeFormat='degree', scale=scale)
        else:
            opts = gdal.DEMProcessingOptions(computeEdges=True, slopeFormat='degree')
        gdal.DEMProcessing(slope_fn, dem_fn, processing="slope", options=opts)
        return 0

    @classmethod
    def dem_to_aspect(cls, dem_fn, aspect_fn):
        opts = gdal.DEMProcessingOptions(computeEdges=True)
        gdal.DEMProcessing(aspect_fn, dem_fn, processing="aspect", options=opts)
        return 0

    @classmethod
    def dem_to_watershed(cls):
        raise NotImplementedError

    @classmethod
    def dem_to_twi(cls):
        raise NotImplementedError

    @classmethod
    def show_map(cls, fn):
        raise NotImplementedError

    @classmethod
    def convert_raster_to_polygon(cls, fn):
        raise NotImplementedError

    @classmethod
    def load_shapefile(cls, fn):
        """
        Load shapefile's shapes and records

        Parameters
        ----------
        fn : str, shp-file file name

        Returns
        -------
        s : shapes, a list of shapes

        r :
        """
        sf = shapefile.Reader(fn)
        s = sf.shapes()
        r = sf.records()
        return s, r

    @classmethod
    def get_raster_epsg(cls, fn):
        """
        Get the EPSG code of the raster data

        Parameters
        ----------
        fn : str, raster file name

        Returns
        -------
        The EPSG code of the raster data, int
        """
        ds = gdal.Open(fn)
        assert isinstance(ds, gdal.Dataset), "{0} is not a valid gdal raster data set".format(fn)
        proj = gdal.Open(fn).GetProjection()
        SRS.ImportFromWkt(proj)
        return int(SRS.GetAttrValue("AUTHORITY", 1))

    @classmethod
    def reproject_raster_by_epsg(cls, src_fn, dst_epsg, dst_fn):
        """
        Reproject the raster data to another projection, based on its own EPSG and a new EPSG

        Parameters
        ----------
        src_fn :

        dst_epsg :

        dst_fn :

        Returns
        -------

        None
        """
        assert isinstance(gdal.Open(src_fn), gdal.Dataset), "{0} is not a valid gdal raster data set".format(src_fn)
        warpopts = gdal.WarpOptions(dstSRS='EPSG:{0}'.format(dst_epsg))
        gdal.Warp(dst_fn, src_fn, options=warpopts)
        return 0

    @classmethod
    def mask_raster(cls, src_fn, mask_shapefile_fn):
        LOGGER.debug("START LOADING RASTER FILE : ".format(os.path.basename(src_fn)))
        # First need to reproject src_fn
        raise NotImplementedError

    @classmethod
    def mask_hdf5(cls, hdf5_fn, mask_shapefile_fn, match_raster_fn, dst_fn,
                  hdf5_arr="SWE", hdf5_lat="lat", hdf5_lon="lon",
                  hdf5_idx=0, buffer_size=5, nodata=-9999.):
        """
        MASK and CLIP the hdf5 file into the extent of the masking shapefile and matching raster file, it
        is going to save the file into the destination filename

        Parameters
        ----------
        hdf5_fn : str, hdf5 file name, the hdf5_fn must have a WGS84 projection (EPSG 4326)

        mask_shapefile_fn : str, masking polygon shapefile

        match_raster_fn : str, matching extent raster file

        dst_fn : str or list(str), saving destination file name(s)

        hdf5_arr : str, hdf5 image array field

        hdf5_lat : str, hdf5 image latitude field

        hdf5_lon : str, hdf5 image longitude field

        hdf5_idx : int or list(int), the index/indices of the first dimension of the hdf5 array

        buffer_size : int, the buffer of the intermediate tmp file

        nodata : float, the nodata value for the hdf5 data set

        Returns
        -------
        """
        # Load the hdf5 image as the specified slice
        LOGGER.debug("START LOADING HDF5 FILE: {0}".format(os.path.basename(hdf5_fn)))
        hdf5_img, hdf5_geotrans = cls.load_hdf5_slices(hdf5_fn, hdf5_idx, hdf5_arr, hdf5_lat, hdf5_lon)
        LOGGER.debug("LOADING HDF5 FILE ({0}) FINISHED!".format(os.path.basename(hdf5_fn)))

        if isinstance(hdf5_idx, list):
            assert isinstance(dst_fn, list)
            assert len(dst_fn) == len(hdf5_idx)
            hdf5_img_sample = hdf5_img[0]
        else:
            hdf5_img_sample = hdf5_img

        # Load the masking polygon
        LOGGER.debug("START MASKING HDF5 FILE")
        s, r = cls.load_shapefile(mask_shapefile_fn)
        assert s[0].shapeType == shapefile.POLYGON, "The shape type is not supported, " \
                                                    "{0}({1}) != {2}({3})".format(s[0].shapeType,
                                                                                  SHAPETYPES[s[0].shapeType],
                                                                                  shapefile.POLYGON,
                                                                                  SHAPETYPES[shapefile.POLYGON])
        assert len(s) == 1, "The number of polygon in the shapefile should be 1, {0} != 1".format(len(s))

        # draw the polygon to an image so the new image is a mask
        latlons = np.array(s[0].points)
        x_idx = np.floor((latlons[:, 0] - hdf5_geotrans[0]) / hdf5_geotrans[1]).astype(int)
        y_idx = np.floor((latlons[:, 1] - hdf5_geotrans[3]) / hdf5_geotrans[5]).astype(int)

        rr, cc = polygon(y_idx, x_idx, hdf5_img_sample.shape)
        polygon_mask = np.zeros_like(hdf5_img_sample)
        polygon_mask[rr, cc] = 1.

        # load the masking raster attributes
        LOGGER.debug("GET MATCHING RASTER (REPROJECT IF NEEDED)")
        tmp_fn = "/tmp/rt_{0}.tif".format(rand_str(12))
        tmp_bool = False
        try:
            rt_match_epsg = cls.get_raster_epsg(match_raster_fn)
        except Exception as e:
            LOGGER.debug(e)
            rt_match_epsg = None
        if rt_match_epsg != 4326:
            cls.reproject_raster_by_epsg(match_raster_fn, 4326, tmp_fn)
            tmp_bool = True
        if tmp_bool:
            rt_match = gdal.Open(tmp_fn, GA_ReadOnly)
        else:
            rt_match = gdal.Open(match_raster_fn, GA_ReadOnly)
        rt_mask_proj = rt_match.GetProjection()
        rt_mask_gt = rt_match.GetGeoTransform()
        rt_mask_xsize = rt_match.RasterXSize
        rt_mask_ysize = rt_match.RasterYSize

        # Clip the hdf5_img to the area we are interested
        LOGGER.debug("GET HDF5 FILE CLIPPING INDICES")
        clip_ul_xidx = np.floor((rt_mask_gt[0] - buffer_size * rt_mask_gt[1] -
                                 hdf5_geotrans[0]) / hdf5_geotrans[1]).astype(int)
        clip_ul_yidx = np.floor((rt_mask_gt[3] - buffer_size * rt_mask_gt[5] -
                                 hdf5_geotrans[3]) / hdf5_geotrans[5]).astype(int)
        clip_lr_xidx = np.floor((rt_mask_gt[0] + (rt_mask_xsize + buffer_size) * rt_mask_gt[1] -
                                 hdf5_geotrans[0]) / hdf5_geotrans[1]).astype(int)
        clip_lr_yidx = np.floor((rt_mask_gt[3] + (rt_mask_ysize + buffer_size) * rt_mask_gt[5] -
                                 hdf5_geotrans[3]) / hdf5_geotrans[5]).astype(int)

        new_gt = (hdf5_geotrans[0] + hdf5_geotrans[1] * clip_ul_xidx,
                  hdf5_geotrans[1],
                  hdf5_geotrans[2],
                  hdf5_geotrans[3] + hdf5_geotrans[5] * clip_ul_yidx,
                  hdf5_geotrans[4],
                  hdf5_geotrans[5])

        if isinstance(hdf5_idx, list):
            idx_list = range(len(hdf5_idx))
            for temp_idx, temp_dst_fn in zip(idx_list, dst_fn):
                temp_hdf5_img = hdf5_img[temp_idx]
                temp_hdf5_img[temp_hdf5_img < nodata] = 0.
                temp_hdf5_img[polygon_mask == 0.] = -9999.

                temp_hdf5_img_clipped = temp_hdf5_img[clip_ul_yidx:clip_lr_yidx, clip_ul_xidx:clip_lr_xidx]

                LOGGER.debug("SAVE CLIPPED RASTER TO TEMPORARY FILE")
                new_tmp_fn = "/tmp/new_rt_{0}.tif".format(rand_str(12))
                cls.write_array_to_raster(new_tmp_fn, temp_hdf5_img_clipped, rt_mask_proj, new_gt)

                LOGGER.debug("REPROJECT RASTER TO DST FILE: {0}".format(os.path.basename(temp_dst_fn)))
                if tmp_bool:
                    cls.reproject_raster(new_tmp_fn, tmp_fn, temp_dst_fn)
                else:
                    cls.reproject_raster(new_tmp_fn, match_raster_fn, temp_dst_fn)

                if os.path.exists(new_tmp_fn):
                    os.remove(new_tmp_fn)

        else:
            hdf5_img[hdf5_img <= nodata] = 0.
            hdf5_img[polygon_mask == 0.] = -9999.

            hdf5_img_clipped = hdf5_img[clip_ul_yidx:clip_lr_yidx, clip_ul_xidx:clip_lr_xidx]

            LOGGER.debug("SAVE CLIPPED RASTER TO TEMPORARY FILE")
            new_tmp_fn = "/tmp/new_rt_{0}.tif".format(rand_str(12))
            cls.write_array_to_raster(new_tmp_fn, hdf5_img_clipped, rt_mask_proj, new_gt)

            LOGGER.debug("REPROJECT RASTER TO DST FILE: {0}".format(os.path.basename(dst_fn)))
            if tmp_bool:
                cls.reproject_raster(new_tmp_fn, tmp_fn, dst_fn)
            else:
                cls.reproject_raster(new_tmp_fn, match_raster_fn, dst_fn)

            if os.path.exists(new_tmp_fn):
                os.remove(new_tmp_fn)

        if os.path.exists(tmp_fn):
            os.remove(tmp_fn)
        return 0
