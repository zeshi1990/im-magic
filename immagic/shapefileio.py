import shapefile
import numpy as np
from raster_utils import SHAPETYPES


class ShapefileDataSetUtils(object):
    pass


class ShapefileDataSet(object):

    def __init__(self, fn, epsg):
        """
        Initialize ShapefileDataSet object
        :param fn:
        :param epsg:
        """
        sf = shapefile.Reader(fn)
        self.shapes = sf.shapes()
        self.records = sf.records()
        self.epsg = epsg

    def select_from_raster(self, rs):
        raise NotImplementedError