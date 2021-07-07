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

from MathTimeFunctions import *
def get_layer(filename, layer):
    '''Return the data from a netcdf file layer
    Parameters
        filename: The filename 
        layers : [Sur_refl_b02_1, QC_250m_1, QC_500m_1, state_1km_1, SensorZenith_1, SolarZenith_1, SolarAzimuth_1,SensorAzimuth_1]
    '''

    hdfdataset = gdal.Open(filename)
    subdataset = hdfdataset.GetSubDatasets()[layer][0]
    bandarray = gdal.Open(subdataset).ReadAsArray()
    return bandarray

def get_layer_tiff(filename):
    '''
    Return an array with value of a geotiff image
    '''
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr


def get_reflectance_layer_tiff(filename):

    '''
    Return an array with reflectance value of Modis a geotiff image
    values < -100 or > 16000 corresponding to modis reflectance invalid value
    0.0001 is the scale factor
    '''
    data = get_layer_tiff(filename)
    mask = (data >= -100) & (data <= 16000)
    data = data * 0.0001
    data[mask == 0] = np.nan
    return data

def get_angle_layer_tiff(filename):

    '''
    Return an array with angles value of Modis a geotiff image
    values < 0 or > 18000 corresponding to modis angles invalid value
    0.01 is the scale factor
    999 corresponding to invalid value.
    '''

    bandarray = get_layer_tiff(filename)
    mask = (bandarray >= 0) & (bandarray <= 18000)
    bandarray = bandarray * 0.01
    bandarray[mask == 0] = 999
    return bandarray


def get_mask(band_data, bit_pos, bit_len, value):
    '''Generates mask with given bit information.
    Parameters
        bit_pos: Position of the specific QA bits in the value string.
        bit_len: Length of the specific QA bits.
        value: A value indicating the desired condition.
    '''
    bitlen = int('1' * bit_len, 2)
    if type(value) == str:
        value = int(value, 2)
    pos_value = bitlen << bit_pos
    print(pos_value)
    con_value = value << bit_pos
    print(con_value)
    mask = (band_data & pos_value) == con_value
    print(mask)
    return mask.astype(int)


def terra_mask(year, idx_row, path, tile):
    '''
    Generate mask with high quality pixels of a day for modis terra images
    Parameters
        idx_row : Day of the year
        path : location of modis reprojected images
        tile : synergy tile
    '''
    date = doy2date(year, int(idx_row)).strftime("%Y%m%d")
    #mod_refl = get_reflectance_layer_tiff(os.path.join(path, 'sur_refl_b02_1/MOD09GQ_sur_refl_b02_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    #qc_mod = get_layer_tiff(os.path.join(path, 'QC_250m_1/MOD09GQ_QC_250m_1_{tile}_{date}_nearest_wgs84.tif').format(tile=tile, date=date)) # nir
    qc_mod = get_layer_tiff(
        os.path.join(path, 'QC_500m_1/MOD09GA_QC_500m_1_{tile}_{date}_nearest_wgs84.tif').format(tile=tile, date=date))
    qa_mod = get_layer_tiff(os.path.join(path, 'state_1km_1/MOD09GA_state_1km_1_{tile}_{date}_nearest_wgs84.tif').format(tile=tile, date=date))

    mask_mod = np.zeros(qc_mod.shape, dtype=np.uint8)
    # band 6 data quality four bit range = "0000" highest quality
    mask_mod += (get_mask(band_data=qc_mod.astype(np.uint16), 
                                  #bit_pos=8, bit_len=4, value='0000') != 1).astype(np.uint8)
                                   bit_pos=22, bit_len=4, value='0000') != 1).astype(np.uint8)

    mask_mod += (get_mask(band_data=qa_mod.astype(np.uint16), 
                                  bit_pos=0, bit_len=2, value='00') != 1).astype(np.uint8)
    mask_mod += (get_mask(band_data=qa_mod.astype(np.uint16), 
                                  bit_pos=2, bit_len=1, value='0') != 1).astype(np.uint8)
    mask_mod += (get_mask(band_data=qa_mod.astype(np.uint16), 
                                  bit_pos=8, bit_len=2, value='00') != 1).astype(np.uint8)
    mask_mod += (get_mask(band_data=qa_mod.astype(np.uint16), 
                                  bit_pos=3, bit_len=3, value='001') != 1).astype(np.uint8)
    mask_mod = mask_mod > 0

    return mask_mod


def aqua_mask(year, idx_row, path, tile):
    '''
    Generate mask with high quality pixels of a day for modis aqua images
    Parameters
        idx_row : Day of the year
        path : location of modis reprojected images
        tile : synergy tile
    '''
    date = doy2date(year, int(idx_row)).strftime("%Y%m%d")
    #myd_refl = get_reflectance_layer_tiff(os.path.join(MODIS_PATH, 'sur_refl_b02_1/MYD09GQ_sur_refl_b02_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    #qc_myd = get_layer_tiff(os.path.join(path, 'QC_250m_1/MYD09GQ_QC_250m_1_{tile}_{date}_nearest_wgs84.tif').format(tile=tile, date=date))
    qc_myd = get_layer_tiff(
        os.path.join(path, 'QC_500m_1/MYD09GA_QC_500m_1_{tile}_{date}_nearest_wgs84.tif').format(tile=tile, date=date))
    qa_myd = get_layer_tiff(os.path.join(path, 'state_1km_1/MYD09GA_state_1km_1_{tile}_{date}_nearest_wgs84.tif').format(tile=tile, date=date))

    mask_myd = np.zeros(qc_myd.shape, dtype=np.uint8)
    # band 6 data quality four bit range = "0000" highest quality
    mask_myd += (get_mask(band_data=qc_myd.astype(np.uint16), 
                                  bit_pos=22, bit_len=4, value='0000') != 1).astype(np.uint8) # highest quality

    mask_myd += (get_mask(band_data=qa_myd.astype(np.uint16), 
                                  bit_pos=0, bit_len=2, value='00') != 1).astype(np.uint8) #no clouds
    mask_myd += (get_mask(band_data=qa_myd.astype(np.uint16), 
                                  bit_pos=2, bit_len=1, value='0') != 1).astype(np.uint8) # no shaddow
    mask_myd += (get_mask(band_data=qa_myd.astype(np.uint16), 
                                  bit_pos=8, bit_len=2, value='00') != 1).astype(np.uint8) # no cirrus
    mask_myd += (get_mask(band_data=qa_myd.astype(np.uint16), 
                                  bit_pos=3, bit_len=3, value='001') != 1).astype(np.uint8)
    mask_myd = mask_myd > 0

    return mask_myd

def olci_mask(year, idx_row, path, tile):
    '''
    Generate mask with high quality pixels of a day for olci images
    Parameters
        idx_row : Day of the year
        path : location of modis reprojected images
        tile : synergy tile
    '''    
    date = doy2date(year, int(idx_row)).strftime("%Y%m%d")
    print("olci_mask : date")
    s3_filename = 'SY_2_SYN-L3-P1D-{tile}-{date}-1.7.nc'.format(tile=tile, date=date)

    s3_ds = nc4.Dataset(os.path.join(path, s3_filename))
    s3_olci_refl = s3_ds['SDR_Oa17'][:]


    olci_flags = s3_ds['OLC_flags'][:] 

    sln_flags = s3_ds['SLN_flags'][:] 
                
    cloud_flags = s3_ds['CLOUD_flags'][:] 
                
    mask = (cloud_flags != 0)# & (olci_flags >= 2**9) & (syn_flags)
    mask_olci = np.zeros(s3_olci_refl.shape, dtype=np.uint8)
    mask_olci = ~np.isnan(s3_olci_refl)

    return mask_olci


def slstr_mask(year, idx_row, path, tile):
    '''
    Generate mask with high quality pixels of a day for modis slstr images
    Parameters
        idx_row : Day of the year
        path : location of modis reprojected images
        tile : synergy tile
    '''
    date = doy2date(year, int(idx_row)).strftime("%Y%m%d")
    s3_filename = 'SY_2_SYN-L3-P1D-{tile}-{date}-1.7.nc'.format(tile=tile, date=date)

    s3_ds = nc4.Dataset(os.path.join(path, s3_filename))
    s3_slstr_refl = s3_ds['SDR_S5N'][:] #SLSTR as Reference
    #s3_slstr_refl = s3_ds['SDR_S3N'][:]

    mask_slstr = np.zeros(s3_slstr_refl.shape, dtype=np.uint8)
    mask_slstr = ~np.isnan(s3_slstr_refl)

    return mask_slstr

