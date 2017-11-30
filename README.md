# *im-magic*
Remote sensing image processing toolkit

### 1. *Introduction*
Researchers working on geospatial data usually struggle with
the various projection, resolution, reference of their spatial
data. Although GIS software can solve these issues. But most of
these softwares are not friendly to use when dealing with batch
processing, and it is kind of difficult to standardize data
protocols among researchers within a research group, which can
make it difficult to work on data produced by different person.

To standardize this process within GlaserLab, we implemented 
this im-magic package to help us standardize our raster data
processing workflow. And the package also allows users to
do raster computations.

### 2. *Features*
This package is made up of . This file contains a class called RasterUtils.
In RasterUtils, we encode functions for loading files, delineation, resample raster data, reprojection, clipping raster, masking rasterdata and masking hdf5;

In this file, we have two classes. The first class is called RasterDataSet.
Here, we define a new class based on GDAL and it inherits some properties of gdal.
You can use self.ds to employ functions defined in GDAL. Also we defined some
algebra calculations for RasterDataSet such as +,-,*,/ and readd and callself.
The second class is called RasterDataSetUtils. This one depends on the class RasterDataSet.
The input for functions in this class requires the RasterDataSet(You can use function called "read_raster" to implement this).
RasterDataSetUtils also includes some other functions to handle the RasterDataSet which are easy to use.
Such as reproject rasterdataset by EPSG, masking raster and mosaic rasterdataset.

For most functions with a return, we usually prepare two cases.
Some of these functions require you to enter the destination filename to be saved as.
We also prepare functions with same function which will return a file saved in memory.

### 3. *Documentation*

#### 1. _raster_utils.py_
##### 1. *RasterUtils*

###### 1. *_load_hdf5_slices_*(fn, hdf5_idx, hdf5_arr="SWE", hdf5_lat="lat", hdf5_lon="lon")
    function propose: get a single slice from a hdf5 dataset
    Parameters
        fn : str, hdf5 file name
             This object should be a string which is the path of a document such as .tif
        hdf5_idx : int, the index of the image slice in the hdf5 data cube
        hdf5_arr : str, hdf5 image array field code
        hdf5_lat : str, hdf5 image array latitude coordinate code
        hdf5_lon : str, hdf5 image array longitude coordinate code
    Returns
        array : The slice image
        geotrans : the image's geotransform.

###### 2. *_load_hdf5_*(fn)
    function purpose: Load a hdf5 xarray data set
    Parameters
        fn : str, hdf5 data set file name.
    Returns
        xarray data set of the hdf5 data

###### 3.*_get_epsg_*(epsg)
    function Purpose : Get detailed projection information for a EPSG code
    Parameters
        epsg : int
    Returns
         str, EPSG detailed information

###### 4.*_delineate_raster_to_polygon_*(src_fn, dst_fn, name, dst_epsg=4326)
    Function Purpose : Delineate the boundary of the raster data file to a polygon and save it as a polygon file
    Parameters
        src_fn : str, source raster filename
        dst_fn : str, filename of the destinated polygon file
        name : str, name of the polygon (for example, if the polygon represents American River basin then the name
               will be american river.
        dst_epsg : int, epsg code
    Returns
        0, saved as polygon file at the destination place

###### 5.*_resample_raster_by_resolution_*(src_fn, dst_fn, x_res, y_res, dst_epsg=None, src_nodata=-9999., dst_nodata=-9999.)
     function purpose : Resample a raster file to destination resolution and destination epsg projection and save it as a raster data file
     Parameters
        src_fn : str, source filename
        dst_fn : str, destination filename
        x_res : float, new x resolution
        y_res : float, new y resolution
        dst_epsg : int, destination epsg code
        src_nodata : float, source raster nodata value
        dst_nodata : float, destination raster nodata value (remember glaserlab protocol requires nodata to be -9999.
     Returns
        0

###### 6.*_calculate_new_geotransform_*(src_gt, ix_y, ix_x)
     function Purpose
        Calculate new geotransform information when clipping the raster image from [ix_y:yyy, ix_x:xxx]
     Parameters
        src_gt : source file geotransform
        ix_y : starting y index
        ix_x : starting x index
     Returns
        a new geotransform tuple

###### 7.*_write_array_to_raster_* (dst_fn, arr, proj, geotransform)
     function purpose
        Write an array to a GeoTIFF file

     Parameters
        dst_fn : str, destination filename to write the array to
        arr : numpy.ndarray, image array
        proj : str, projection in wkt format
        geotransform : tuple, geotransform tuple, len(geotransform) = 6

     Returns
        0

###### 8a.*_reproject_raster_*(src_fn, match_fn, dst_fn, gra_type=GRA_Bilinear)
     function purpose
        Reproject a raster with reference of a matching raster data and the result will be saved in a GeoTIFF file
     Parameters
        src_fn : str, source filename to reproject from
        match_fn : str, match filename that the reprojected file's projection and geotransform will match with it
        dst_fn : str, destination filename that the raster will save to
        gra_type : int, gdal.GRA_***
     Returns
        0

###### 8b.*_reproject_raster_to_mem_* (src, match, gra_type=GRA_Bilinear)
     function purpose
        Reproject a raster with reference of a matching raster data
     Parameters
        src_fn : str, source filename to reproject from
        match_fn : str, match filename that the reprojected file's projection and geotransform will match with it
        gra_type : int, gdal.GRA_***
     Returns
        a raster dataset

###### 9a.*_clip_raster_*(src_fn, dst_fn, ulx, uly, lrx, lry)
     function purpose
        Clip a raster based on its upperleft coord and lowerright coord
     Parameters
        src_fn : str, the filename of the raster data to be clipped
        dst_fn : str, the filename of clipped raster you want to save as
        ulx : float, upperleft x, remember longitude is x
        uly : float, upperleft y, remember latitude is y
        lrx : float, lowerright x
        lry : float, lowerright y

     Returns
        0

###### 9b.*_clip_raster_by_idx_*(src_fn, dst_fn, ulx_idx, uly_idx, lrx_idx, lry_idx)
     function purpose
        Clip a raster based on its upperleft coord and lowerright coord
     Parameters
        src_fn : str, the filename of the raster data to be clipped
        dst_fn : str, the filename of clipped raster data you want to save as
        ulx : float, upperleft x, remember longitude is x
        uly : float, upperleft y, remember latitude is y
        lrx : float, lowerright x
        lry : float, lowerright y

     Returns
        0

###### 10.*_dem_to_slope_*(dem_fn, slope_fn, scale=None)
      Processing DEM and convert it to slope

      Parameters
        dem_fn :
        slope_fn :
        scale :

      Returns
        None

###### 11.*_dem_to_aspect_*(dem_fn, aspect_fn):
      Processing DEM and convert it to aspect

      Parameters
        dem_fn :
        aspect_fn :

      Returns
        None

###### 12.*_dem_to_watershed_*(n, l_points, dst_point_shape, src_dem_fn, snap_dist, flow_thre, dst_watershed_fn, dst_points_fn, dst_stream_fn):

      Processing DEM and Delineate the watershed out
      Parameters
        n: int, the number of outlet ponits of water
        l_points: a list of coordinates, list of points that you think could be the outlet points of water
                         every point has four values(x,y,z,w)
                         x,y:coordinates, z:elevation w:measure values if you don't have z and w, just enter 0
        dst_watershed_fn: a string, the name of shapefile of water's outlet points
        src_dem_fn: a string, the file name of the DEM file to be handled. the file format should be tiff
        dst_watershed_fn: a string,
        dst_points_fn: a string,
        dst_stream_fn: a string,
      Return
        none

###### 13.*_dem_to_twi_*(src_fn):

        Calculate the twi for an area
        Parameters
          src_fn: a string, the filename of the DEM file
        Return
          an array matrix, the twi of an area

###### 14.*_dem_to_twi_file_*(src_fn, dst_fn):

        calculate the twi for an area and save the result as a file object
        Parameters
          src_fn: a string, the filename of the DEM
          dst_fn: a string, the file name of the file that twi is goint to be saved in
        Return
          none

###### 15.*_show_map_*(fn)
        Plot the image out
        Parameters
          fn:the flename of the image to plot

###### 10.*_reset_raster_nodata_*(src_fn, src_nodata)
     function purpose
        Reset raster data's nodata

     Parameters
        ----------
        src_fn : str,
        src_nodata :

     Returns
        -------
        0

###### 11.*_load_shapefile_*(cfn, fields=False)
     function purpose
        Load shapefile's shapes and records

     Parameters
        ----------
        fn : str, shp-file file name you want to load
        fields : bool, include fields in results or not

     Returns
        -------
        s : shapes, a list of shapes
        r : records, a list of json records

###### 12.*_get_raster_epsg_*(fn)
     function purpose
        Get the EPSG code of the raster data

     Parameters
        ----------
        fn : str, file name of raster dataset

     Returns
        -------
        The EPSG code of the raster data, int

###### 13.*_reproject_raster_by_epsg_*(src_fn, dst_epsg, dst_fn)

     function purpose
        Reproject the raster data to another projection, based on its own EPSG and a new EPSG

     Parameters
        ----------
        src_fn : str, source filename, you know what does it mean
        dst_epsg : int, integer
        dst_fn : str, destination filename, you know what does it mean
     returns
        0

###### 14a.*_mask_raster_*(src_fn, mask_shapefile_fn, match_raster_fn, dst_fn = None)

      function purpose: mask the raster data
      Parameters
         src_fn : str, raster data file name
         mask_shapefile_fn : str, shapefile file name
         match_raster_fn : str, name of raster data corresponding to shapefile
         dst_fn : the detination file name of the output raster data
      Returns
         case1 : if don't enter dst_fn, then the return will be a raster dataset
         case2 : if we enter a dst_fn, then the function will form a file of rasterdata at destinated place

###### 14b.

###### 15.*_mask_hdf5_*(hdf5_fn, mask_shapefile_fn, match_raster_fn, dst_fn, hdf5_arr="SWE", hdf5_lat="lat", hdf5_lon="lon", hdf5_idx=0, buffer_size=5, nodata=-9999.):


      fucntion purpose
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
        0


#### 2. *rasterio.py*
##### 1. *_Class: RasterDataSetUtils_*
###### 1. *_read_raster_*
        function purpose: read a raster file as raster dataset class so that we can use the property we defined in raster data set class
        parameters
           fn: str, the location and name of the raster file to be read
        Returns
           ds: a rasterdataset

###### 2. *_reproject_raster_epsg_*
        function purpose:
        parameters

        Returns

###### 3. *_mask_raster_*
        function purpose:
        parameters

        Returns

###### 4. *_reproject_raster_to_mem_rd_*
        Faunction Purpose
           Reproject a raster with reference of a matching raster data
        Parameters
            src_fn : str, source filename to reproject from
            match_fn : str, match filename that the reprojected file's projection and geotransform will match with it
            gra_type : int, gdal.GRA_***
        Returns
            gdal.Dataset

###### 5. *_mosaic_rasters_*
        function purpose:
        Merge a few rasters together

        Parameters
        l_rasters: raster dataset , the raster data that to be merged together

        Returns
        mosaiced_ds: the mosaiced image



##### 2. *_RasterDataSet_*
###### 1. *_"+"_*: self *_+_* other
        *function purpose*: add a rasterdataset or a numbertype data to a rasterdataset data
        *parameters*
           self: rasterdataset
                          The object before *_+_* must be a rasterdataset data type
           other: rasterdataset or numbertype data
                           The object after *_+_* must be a rasterdataset or a numbertype data set


###### 2. *_"-"_*: self *_-_* other
        *function purpose*: subtract a rasterdataset or a numbertype data to a rasterdataset data
        *parameters*
                   *self: rasterdataset
                          The object before *_-_* must be a rasterdataset data type
                   *other: rasterdataset or numbertype data
                           The object after *_-_* must be a rasterdataset or a numbertype data set
        Return

###### 3. "*": self * other
        *function purpose*: multiply a rasterdataset or a numbertype data to a rasterdataset data
        *parameters*
                   *self: rasterdataset
                          The object before *_*_* must be a rasterdataset data type
                   *other: rasterdataset or numbertype data
                           The object after *_*_* must be a rasterdataset or a numbertype data set
        Return


###### 4. "/": self / other
        function purpose: multiply a rasterdataset or a numbertype data to a rasterdataset data
        *parameters*
                   *self: rasterdataset
                          The object before *_*_* must be a rasterdataset data type
                   *other: rasterdataset or numbertype data
                           The object after *_*_* must be a rasterdataset or a numbertype data set
        Return

###### 5. ""
        function purpose
          readd a raster dataset by itself
        Parameters
          rasterdataset
        Returns
          the rasterdata set add by itself

### 4. *License*

### 5. *Cite our software*
