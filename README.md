# im-magic
Remote sensing image processing toolkit

### 1. Introduction
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

### 2. Features

### 3. Documentation

###### 1. *_load_hdf5_slices_*(fn, hdf5_idx, hdf5_arr="SWE", hdf5_lat="lat", hdf5_lon="lon")
<pre>
<i>function propose</i>: get a single slice from a hdf5 dataset
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
</pre>

### 4. License

### 5. Cite our software