import os
import pandas as pd
import datetime
from fiona.crs import from_epsg
import numpy as np
import gdal
import datetime
import fiona
import rasterio as rio
import rasterio.merge
import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
import os
import glob
import pickle
import datetime
import numpy as np
import pandas as pd
import gdal
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
import netCDF4 as nc4



def array_to_dict(key, value):
    dictionary = {}
    for A, B in zip(key, value):
        dictionary[A] = B
    del dictionary['doy']
    return dictionary


def doy2date(year, days):
    return datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1)


def date2doy(date):
    return (date - datetime.datetime(date.year, 1, 1)).days + 1


