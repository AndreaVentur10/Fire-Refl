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
from kernels import *
def invert_composite(qa, rho, kern):
    """A function to invert reflectance assuming the kernels in `kern`,
    `qa` holds the QA data, `doys` the dates, and `rho` the actual
    observations. Will composite all observations between `doy_min` and
    `doy_max`."""
    passer = qa
    obs = rho[passer]
    K = kern[passer, :]
    (f, rmse, rank, svals) = np.linalg.lstsq(K, obs, rcond=None)
    fwd = K.dot(f.T)
    return obs, fwd, K, passer, f, rmse

def brdf_approach(i, window, location , water, tile, year, band = 'nir', output_shape = (3600, 3600)):
    '''
    Generate merged image with brdf normalization approach
    parameters : 
        i : day of the year
        window ; brdf window
        location : merged images location
        water : water mask
    '''
    init = i - window//2
    middle = init + window//2
    end = i + window//2
                                        
    if not os.path.exists(location + 'brdf/brdf_merged_random_forest_{tile}_{band}_{year}_{day}.npz'.format(year = year , tile = tile , band = band, day = middle)):   
        refl = []
        sza_all = []
        saa_all = []
        vza_all = []
        vaa_all = []
        im = np.zeros(output_shape)
        errors = np.zeros(output_shape)

        for j in range(init, end + 1):
            #print(int(j))
            ds = np.load(location + 'images/merged_random_forest_{tile}_{band}_{year}_{day}.npz'.format(year = year , tile = tile ,  band = band, day = j))
            refl.append(ds['refl'])
            sza_all.append(ds['sza'])
            saa_all.append(ds['saa'])
            vza_all.append(ds['vza'])
            vaa_all.append(ds['vaa'])
        refl = np.array(refl)
        sza_angles = np.array(sza_all)
        saa_angles = np.array(saa_all) 
        vza_angles = np.array(vza_all) 
        vaa_angles = np.array(vaa_all)

        for k in range(output_shape[0]): #3600
            for l in range(output_shape[1]):  #3600          
                time_series = refl[:, k, l]
                #print(time_series)
                sza = sza_angles[:, k, l]
                saa = saa_angles[:, k, l]
                vza = vza_angles[:, k, l]
                vaa = vaa_angles[:, k, l]

                if water[k, l]:
                    im[k, l] = refl[time_series.shape[0]//2, k, l]
                    #print('c')
                    continue

                raa = vaa - saa
                qa = ~np.isnan(time_series)

                if not qa[time_series.shape[0]//2]:
                    im[k, l] = np.nan
                    errors[k, l] = np.nan
                    continue

                indexes = np.arange(time_series.shape[0])
                indexes = indexes[qa]

                if np.sum(qa) >= 3:
                    # Generate the kernels,
                    K_obs = Kernels(vza, sza, raa,
                                    LiType='Sparse', doIntegrals=False,
                                    normalise=1, RecipFlag=True, RossHS=False, MODISSPARSE=True,
                                    RossType='Thick')
                    n_obs = time_series.shape[0]
                    kern = np.ones((n_obs, 3))  # Store the kernels in an array
                    kern[:, 1] = K_obs.Ross
                    kern[:, 2] = K_obs.Li

                    obs, fwd, K, passer, f, rmse = invert_composite(qa, time_series, kern)
                    
                    im[k, l] = fwd[indexes == time_series.shape[0] // 2]

                    if len(rmse) > 0:
                        errors[k, l] = rmse[0]

                    else:
                        errors[k, l] = np.nan 
                else:
                    im[k, l] = np.nan
                    errors[k, l] = np.nan
        np.savez_compressed(location + 'brdf/brdf_merged_random_forest_{tile}_{band}_{year}_{day}.npz'.format(year = year , tile = tile , band = band, day = middle), 
                            refl=im, rmse=errors)