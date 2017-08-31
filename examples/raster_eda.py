import gdal
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat

from immagic import RasterUtils as RU

# fn = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/Tuolumne_2014_bareDEM_3p0m_agg_EXPORT.tif"
# RU.reset_raster_nodata(fn, 0.)

fn = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/fr_data/bkl/slope.tif"
new_fn = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/fr_data/bkl_new/slope.tif"
ds = gdal.Open(fn)
print ds.GetProjection()
print ds.GetGeoTransform()
new_ds = gdal.Open(new_fn)
print new_ds.GetProjection()
print new_ds.GetGeoTransform()


sites = ["bkl", "grz", "hbg", "ktl"]
features = ["dem", "slope", "aspect", "canopy"]
fn_pattern = "/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/fr_data/{0}_new/{1}.tif"
features_dict = {}
for site in sites:
    feature_arr = None
    for feature in features:
        ds = gdal.Open(fn_pattern.format(site, feature))
        if feature_arr is None:
            feature_arr = ds.ReadAsArray().flatten()
        else:
            feature_arr = np.column_stack((feature_arr, ds.ReadAsArray().flatten()))
    features_dict[site] = feature_arr
savemat("/Users/zeshizheng/Google Drive/dev/im-magic/data/rasters/fr_data/features.mat", features_dict)

fig, axarr = plt.subplots(nrows=2, ncols=1)
axarr[0].imshow(ds.ReadAsArray())
axarr[1].imshow(new_ds.ReadAsArray())
plt.show()
