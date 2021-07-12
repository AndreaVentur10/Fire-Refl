import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
import netCDF4 as nc4
import matplotlib.pyplot as plt

from MathTimeFunctions import *
from LayerFunctions import *


def random_forest_models(year, idx_row, tile, s3_path, modis_reprojected, outputModel, DemPath, LC, band='red'):
    '''
    The function run random forest model,  save the model, test set and train set
    The function run SLSTR-Terra, SLSTR-Aqua and Terra-Aqua
    Parameters
        idx_row : day of the year
        s3_file : path of synergy files
        modis_reprojected : location of the modis reprojected images
        outputModel : location of the output models
    '''

    dem = get_layer_tiff(os.path.join(DemPath, 'dem.tif'))

    # Invalid data is : -9999
    aspect = get_layer_tiff(os.path.join(DemPath, 'aspect.tif'))
    # Invalid data is : -9999
    slope = get_layer_tiff(os.path.join(DemPath, 'slope.tif'))
    # Invalid data is : 0
    hillshade = get_layer_tiff(os.path.join(DemPath, 'hillshade.tif'))

    landcover = get_layer_tiff(os.path.join(DemPath, LC))

    # header of csv's for the error record of the sensors
    error = pd.DataFrame(columns=['doy', 'Terra', 'Aqua', 'SLSTR'])
    error1 = pd.DataFrame(columns=['doy', 'Aqua'])

    date = doy2date(year, int(idx_row)).strftime("%Y%m%d")
    # Load SLSTR from Synergy file
    s3_filename = 'SY_2_SYN-L3-P1D-{tile}-{date}-1.7.nc'.format(tile=tile, date=date)

    s3_ds = nc4.Dataset(os.path.join(s3_path, s3_filename))
    # Load SLSTR data: S5 SWIR Band , Wavelength(µm): 1.61
    s3_slstr_refl = s3_ds['SDR_S5N'][:]
    # fill the row for the record errors (in case there is info in the reference) with nans
    error.loc[len(error)] = [np.nan, np.nan, np.nan, np.nan]
    # mask slstr
    s3_mask_slstr = slstr_mask(year, idx_row, s3_path, tile)

    mod_refl = get_reflectance_layer_tiff(os.path.join(modis_reprojected,
                                                       'sur_refl_b06_1/MOD09GA_sur_refl_b06_1_{tile}_{'
                                                       'date}_nearest_wgs84.tif'.format(
                                                           tile=tile, date=date)))  # change
    # mask terra
    mod_mask = terra_mask(year, idx_row, modis_reprojected, tile)
    myd_refl = get_reflectance_layer_tiff(os.path.join(modis_reprojected,
                                                       'sur_refl_b06_1/MYD09GA_sur_refl_b06_1_{tile}_{'
                                                       'date}_nearest_wgs84.tif'.format(
                                                           tile=tile, date=date)))
    myd_mask = aqua_mask(year, idx_row, modis_reprojected, tile)

    mod_vza = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SensorZenith_1/MOD09GA_SensorZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))
    mod_vaa = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SensorAzimuth_1/MOD09GA_SensorAzimuth_1_{tile}_{date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))
    mod_sza = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SolarZenith_1/MOD09GA_SolarZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))
    mod_saa = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SolarAzimuth_1/MOD09GA_SolarAzimuth_1_{tile}_{date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))

    myd_vza = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SensorZenith_1/MYD09GA_SensorZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))
    myd_vaa = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SensorAzimuth_1/MYD09GA_SensorAzimuth_1_{tile}_{date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))
    myd_sza = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SolarZenith_1/MYD09GA_SolarZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))
    myd_saa = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SolarAzimuth_1/MYD09GA_SolarAzimuth_1_{tile}_{date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))

    # FIRST CONDITION: IF THERE IS DATA FOR THE REFERENCE(SLSTR) THEN
    if (s3_slstr_refl.mask == False).sum() > 0:  # SLSTR as reference

        # slstr-terra: calculate intersection, pixels with high quality in terra and slstr
        intersection = (s3_mask_slstr.data == True) & (mod_mask == False)
        # fill the attributes
        df = pd.DataFrame()
        df['intersection'] = intersection.reshape(-1)
        df['land_cover'] = landcover.reshape(-1)
        df['dem'] = dem.reshape(-1)  # invalid is -214748
        df['slope'] = slope.reshape(-1)  # invalid is -9999
        df['aspect'] = aspect.reshape(-1)  # invalid -9999
        df['hillshade'] = hillshade.reshape(-1)  # invalid 0
        # terra sensors
        df['vza'] = mod_vza.reshape(-1)
        df['vaa'] = mod_vaa.reshape(-1)
        df['sza'] = mod_sza.reshape(-1)
        df['saa'] = mod_saa.reshape(-1)

        df['terra'] = mod_refl.reshape(-1)
        df['slstr'] = s3_slstr_refl.reshape(-1)  # SLSTR as reference: put as a column
        df = df[df.intersection == True]
        # delete invalid values
        df = df[df.dem > -214748]
        df = df[df.slope > -9999]
        df = df[df.aspect > -9999]
        df = df.drop('intersection', axis=1)
        l = df['land_cover'].value_counts(sort=True)
        # delete if there is a land cover with only one pixel
        for i in range(0, len(l)):
            if l.iloc[-1] < 2:
                print('clean ' + str(list(l.keys())[-1]))
                df = df[df['land_cover'] != list(l.keys())[-1]]
                l = df['land_cover'].value_counts(sort=True)

        # prepare the data: training and testing sets
        if len(df) > 1:
            # split test & train based on the land cover and save them
            df['label'] = df.index
            df.reset_index(drop=True, inplace=True)

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            for train_index, test_index in split.split(df, df['land_cover']):
                strat_train_set = df.loc[train_index]
                strat_test_set = df.loc[test_index]

            train = strat_train_set
            test = strat_test_set

            # run the model  and save it
            train = train.drop('label', axis=1)  # delete label column from train set
            test = test.drop('label', axis=1)  # delete label column from test set
            y_train = np.array(train['slstr'])  # SLSTR as reference
            y_test = np.array(test['slstr'])  # SLSTR as reference

            # drop the terra
            train = train.drop('slstr', axis=1)  # SLSTR as reference
            test = test.drop('slstr', axis=1)  # SLSTR as reference

            X_train = train
            X_test = test

            #### START TERRA MODEL ####
            # Random forest model
            regressor = RandomForestRegressor(max_depth=15, n_estimators=50, random_state=0)
            # Fit the model
            regressor.fit(X_train, y_train)
            # save the model
            pkl_name = os.path.join(outputModel,
                                    'SLSTR-Terra/random_forest_SLSTR-Terra_{tile}_{band}_{year}_{day}.pkl').format(
                tile=tile, band=band, day=idx_row, year=year)
            with open(pkl_name, 'wb') as file:
                pickle.dump(regressor, file)
            # Validation of the model
            y_pred = regressor.predict(test)
            # write the error score
            error.loc[len(error) - 1]['doy'] = idx_row
            error.loc[len(error) - 1]['Terra'] = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        # slstr-aqua: calculate intersection, pixels with high quality in aqua and slstr
        intersection = (s3_mask_slstr.data == True) & (myd_mask == False)
        # fill the attributes
        df = pd.DataFrame()
        df['intersection'] = intersection.reshape(-1)
        df['land_cover'] = landcover.reshape(-1)
        df['dem'] = dem.reshape(-1)  # invalid is -214748
        df['slope'] = slope.reshape(-1)  # invalid is -9999
        df['aspect'] = aspect.reshape(-1)  # invalid -9999
        df['hillshade'] = hillshade.reshape(-1)  # invalid 0
        # terra sensors
        df['vza'] = myd_vza.reshape(-1)
        df['vaa'] = myd_vaa.reshape(-1)
        df['sza'] = myd_sza.reshape(-1)
        df['saa'] = myd_saa.reshape(-1)

        df['aqua'] = myd_refl.reshape(-1)
        df['slstr'] = s3_slstr_refl.reshape(-1)
        df = df[df.intersection == True]
        df = df[df.dem > -214748]
        df = df[df.slope > -9999]
        df = df[df.aspect > -9999]
        df = df.drop('intersection', axis=1)
        l = df['land_cover'].value_counts(sort=True)
        # delete if there is land cover with one pixel 
        for i in range(0, len(l)):
            if l.iloc[-1] < 2:
                print('clean ' + str(list(l.keys())[-1]))
                df = df[df['land_cover'] != list(l.keys())[-1]]
                l = df['land_cover'].value_counts(sort=True)
        # prepare the data #save test and train set
        if len(df) > 1:
            # split test & train and save them
            df['label'] = df.index
            df.reset_index(drop=True, inplace=True)

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            for train_index, test_index in split.split(df, df['land_cover']):
                strat_train_set = df.loc[train_index]
                strat_test_set = df.loc[test_index]

            train = strat_train_set
            test = strat_test_set
            # run the model  and save it
            train = train.drop('label', axis=1)
            test = test.drop('label', axis=1)

            y_train = np.array(train['slstr'])  # SLSTR as reference
            y_test = np.array(test['slstr'])  # SLSTR as reference

            # drop the terra
            train = train.drop('slstr', axis=1)  # SLSTR as reference
            test = test.drop('slstr', axis=1)  # SLSTR as reference

            X_train = train
            X_test = test

            #### START AQUA MODEL ####
            regressor = RandomForestRegressor(max_depth=15, n_estimators=50, random_state=0)
            # fit the model
            regressor.fit(X_train, y_train)
            # save this model
            pkl_name = os.path.join(outputModel,
                                    'SLSTR-Aqua/random_forest_SLSTR-Aqua_{tile}_{band}_{year}_{day}.pkl').format(
                tile=tile, band=band, day=idx_row, year=year)
            with open(pkl_name, 'wb') as file:
                pickle.dump(regressor, file)

            y_pred = regressor.predict(test)
            error.loc[len(error) - 1]['doy'] = idx_row
            error.loc[len(error) - 1]['Aqua'] = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    else:  # No slstr data or not enough
        # Terra is the reference and aqua
        # Train model between terra and Aqua

        intersection = (myd_mask == False) & (mod_mask == False)

        df = pd.DataFrame()
        df['intersection'] = intersection.reshape(-1)
        df['land_cover'] = landcover.reshape(-1)
        df['dem'] = dem.reshape(-1)  # invalid is -214748
        df['slope'] = slope.reshape(-1)  # invalid is -9999
        df['aspect'] = aspect.reshape(-1)  # invalid -9999
        df['hillshade'] = hillshade.reshape(-1)  # invalid 0
        # terra sensors
        df['vza'] = myd_vza.reshape(-1)
        df['vaa'] = myd_vaa.reshape(-1)
        df['sza'] = myd_sza.reshape(-1)
        df['saa'] = myd_saa.reshape(-1)

        df['aqua'] = myd_refl.reshape(-1)
        df['terra'] = mod_refl.reshape(-1)
        df = df[df.intersection == True]
        df = df[df.dem > -214748]
        df = df[df.slope > -9999]
        df = df[df.aspect > -9999]
        df = df.drop('intersection', axis=1)
        l = df['land_cover'].value_counts(sort=True)

        for i in range(0, len(l)):
            if l.iloc[-1] < 2:
                print('clean ' + str(list(l.keys())[-1]))
                df = df[df['land_cover'] != list(l.keys())[-1]]
                l = df['land_cover'].value_counts(sort=True)

        # prepare the data #save test and train set devide
        if len(df) > 1:
            # split test & train and save them
            df['label'] = df.index
            df.reset_index(drop=True, inplace=True)

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            for train_index, test_index in split.split(df, df['land_cover']):
                strat_train_set = df.loc[train_index]
                strat_test_set = df.loc[test_index]

            train = strat_train_set
            test = strat_test_set
            # run the model  and save it
            train = train.drop('label', axis=1)
            test = test.drop('label', axis=1)
            y_train = np.array(train['terra'])
            y_test = np.array(test['terra'])

            # drop the terra
            train = train.drop('terra', axis=1)
            test = test.drop('terra', axis=1)

            X_train = train
            X_test = test
            print('start terra aqua model')

            regressor = RandomForestRegressor(max_depth=15, n_estimators=50, random_state=0)
            regressor.fit(X_train, y_train)
            print("finish model")

            pkl_name = os.path.join(outputModel,
                                    'Terra-Aqua/random_forest_Terra-Aqua_{tile}_{band}_{year}_{day}.pkl').format(
                tile=tile, band=band, day=idx_row, year=year)
            with open(pkl_name, 'wb') as file:
                pickle.dump(regressor, file)
            y_pred = regressor.predict(test)
            error1.loc[len(error1)] = [idx_row, np.sqrt(metrics.mean_squared_error(y_test, y_pred))]

    key = np.array(error.columns)
    rank = pd.DataFrame(columns=["1", "2", "3"])
    for i in range(0, len(error)):
        t = [0] * 3
        j = 0
        l = np.array(error[i:i + 1]).reshape(4, -1)
        dictionary = array_to_dict(key, l)
        dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}
        for ke in dictionary:
            if np.isnan(dictionary[ke][0]):
                t[j] = np.nan
                j = j + 1
            else:
                t[j] = ke
                j = j + 1
        rank.loc[len(rank)] = t

    error.to_csv(os.path.join(outputModel, 'error_sensor_random_forest.csv'), index=False)
    error1.to_csv(os.path.join(outputModel, 'error_aqua.csv'), index=False)
    rank.to_csv(os.path.join(outputModel, 'ranking_sensors_random_forest.csv'), index=False)


def random_forest(year, idx_row, tile, s3_path, modis_reprojected, outputModel, DemPath, LC, band='red'):
    '''
    The function generate joined image
    Parameters
        idx_row : day of the year
        s3_file : path of synergy files
        modis_reprojected : location of the modis reprojected images
        outputModel : location of the output image
    '''
    dem = get_layer_tiff(os.path.join(DemPath, 'dem.tif'))
    # Invalid data is : -9999
    aspect = get_layer_tiff(os.path.join(DemPath, 'aspect.tif'))
    # Invalid data is : -9999
    slope = get_layer_tiff(os.path.join(DemPath, 'slope.tif'))
    # Invalid data is : 0
    hillshade = get_layer_tiff(os.path.join(DemPath, 'hillshade.tif'))

    landcover = get_layer_tiff(os.path.join(DemPath, LC))

    watermask = np.where(landcover == 210, 0, 1)
    idx_row = int(idx_row)

    # Load the refl and angles
    rank = pd.read_csv(outputModel + 'ranking_sensors_random_forest.csv')
    date = doy2date(year, int(idx_row)).strftime("%Y%m%d")
    # Load the SLSTR data
    s3_filename = 'SY_2_SYN-L3-P1D-{tile}-{date}-1.7.nc'.format(tile=tile, date=date)
    s3_ds = nc4.Dataset(os.path.join(s3_path, s3_filename))
    s3_slstr_refl = s3_ds['SDR_S5N'][:]  # SLSTR as Reference

    # Get Terra, Aqua & Mask
    mod_refl = get_reflectance_layer_tiff(os.path.join(modis_reprojected, 'sur_refl_b06_1/MOD09GA_sur_refl_b06_1_{'
                                                                          'tile}_{date}_nearest_wgs84.tif'.format(
        tile=tile, date=date)))
    mod_mask = terra_mask(year, idx_row, modis_reprojected, tile)
    myd_refl = get_reflectance_layer_tiff(os.path.join(modis_reprojected, 'sur_refl_b06_1/MYD09GA_sur_refl_b06_1_{'
                                                                          'tile}_{date}_nearest_wgs84.tif'.format(
        tile=tile, date=date)))
    myd_mask = aqua_mask(year, idx_row, modis_reprojected, tile)
    # angles
    mod_vza = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SensorZenith_1/MOD09GA_SensorZenith_1_{tile}_{'
                                                                   'date}_nearest_wgs84.tif'.format(tile=tile,
                                                                                                    date=date)))
    mod_vaa = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SensorAzimuth_1/MOD09GA_SensorAzimuth_1_{tile}_{'
                                                'date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))
    mod_sza = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SolarZenith_1/MOD09GA_SolarZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))
    mod_saa = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SolarAzimuth_1/MOD09GA_SolarAzimuth_1_{tile}_{date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))

    myd_vza = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SensorZenith_1/MYD09GA_SensorZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))
    myd_vaa = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SensorAzimuth_1/MYD09GA_SensorAzimuth_1_{tile}_{'
                                                'date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))
    myd_sza = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SolarZenith_1/MYD09GA_SolarZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))
    myd_saa = get_angle_layer_tiff(os.path.join(modis_reprojected,
                                                'SolarAzimuth_1/MYD09GA_SolarAzimuth_1_{tile}_{date}_nearest_wgs84.tif'.format(
                                                    tile=tile, date=date)))

    s3_saa = s3_ds['SAA'][:]
    s3_sza = s3_ds['SZA'][:]

    sza = np.zeros(s3_slstr_refl.shape)
    saa = np.zeros(s3_slstr_refl.shape)
    vza = np.zeros(s3_slstr_refl.shape)
    vaa = np.zeros(s3_slstr_refl.shape)

    # if slstr info exist, then it's the reference
    if (s3_slstr_refl.mask == False).sum() > 0:  # SLSTR as Reference
        # fill with reference
        tmp = s3_slstr_refl.copy()

        im_subs = np.zeros(s3_slstr_refl.shape, dtype=np.uint8)
        im_subs[~np.isnan(s3_slstr_refl)] = 1  # SLSTR take value 1 in pseudocode
        np.save('intersection1.npy', im_subs)
        # put angles
        sza[~np.isnan(s3_slstr_refl)] = s3_sza[~np.isnan(s3_slstr_refl)]  # SLSTR as Reference
        saa[~np.isnan(s3_slstr_refl)] = s3_saa[~np.isnan(s3_slstr_refl)]  # SLSTR as Reference
        # Load SLSTR - Terra
        filename = os.path.join(outputModel,
                                'SLSTR-Terra/random_forest_SLSTR-Terra_{tile}_{band}_{year}_{day}.pkl'.format(
                                    tile=tile, band=band, year=year, day=idx_row))  # SLSTR as Reference
        # If SLSTR-Terra model exist then
        if os.path.exists(filename):
            terra_model = pickle.load(open(filename, 'rb'))
            d = pd.DataFrame()  # terra
            d['land_cover'] = landcover.reshape(-1)  # invalid is 0
            d['dem'] = dem.reshape(-1)  # invalid is -214748
            d['slope'] = slope.reshape(-1)  # invalid is -9999
            d['aspect'] = aspect.reshape(-1)  # invalid -9999
            d['hillshade'] = hillshade.reshape(-1)  # invalid 0
            d['vza'] = mod_vza.reshape(-1)
            d['vaa'] = mod_vaa.reshape(-1)
            d['sza'] = mod_sza.reshape(-1)
            d['saa'] = mod_saa.reshape(-1)
            d['terra'] = mod_refl.reshape(-1)
            d = d.fillna(0)
            d = np.array(d)
            # predict terra with the SLSTR-Terra model
            predicted_terra = terra_model.predict(d)
            predicted_terra = predicted_terra.reshape(-1, 1)
            predicted_terra = predicted_terra.reshape(3600, 3600)

        # Load SLSTR - Aqua
        filename = os.path.join(outputModel,
                                'SLSTR-Aqua/random_forest_SLSTR-Aqua_{tile}_{band}_{year}_{day}.pkl'.format(
                                    tile=tile, band=band, year=year, day=idx_row))  # SLSTR as Reference
        # If SLSTR-Terra model exist then
        if os.path.exists(filename):
            aqua_model = pickle.load(open(filename, 'rb'))
            d = pd.DataFrame()  # aqua
            d['land_cover'] = landcover.reshape(-1)  # invalid is 0
            d['dem'] = dem.reshape(-1)  # invalid is -214748
            d['slope'] = slope.reshape(-1)  # invalid is -9999
            d['aspect'] = aspect.reshape(-1)  # invalid -9999
            d['hillshade'] = hillshade.reshape(-1)  # invalid 0
            d['vza'] = myd_vza.reshape(-1)
            d['vaa'] = myd_vaa.reshape(-1)
            d['sza'] = myd_sza.reshape(-1)
            d['saa'] = myd_saa.reshape(-1)
            d['aqua'] = myd_refl.reshape(-1)
            d = d.fillna(0)
            d = np.array(d)
            # predict terra with the SLSTR-Aqua model
            predicted_aqua = aqua_model.predict(d)
            predicted_aqua = predicted_aqua.reshape(-1, 1)
            predicted_aqua = predicted_aqua.reshape(3600, 3600)

        # If the first score was from Terra then
        if (rank.loc[0][0] == "Terra") & (os.path.exists(os.path.join(outputModel,
                                                                      'SLSTR-Terra/random_forest_SLSTR-Terra_{tile}_{'
                                                                      'band}_{year}_{day}.pkl').format(
            tile=tile, band=band, year=year, day=idx_row))):
            indexes = (np.isnan(tmp)) & (mod_mask == False)
            tmp[indexes] = predicted_terra[indexes]

            np.save('intersection2.npy', im_subs)
            im_subs[indexes] = 2  # terra id is "2" the final product

            sza[indexes] = mod_sza[indexes]
            saa[indexes] = mod_saa[indexes]
            vza[indexes] = mod_vza[indexes]
            vaa[indexes] = mod_vaa[indexes]
        # If the first score was from Aqua then
        if (rank.loc[0][0] == "Aqua") & (os.path.exists(
                os.path.join(outputModel, 'SLSTR-Aqua/random_forest_SLSTR-Aqua_{tile}_{band}_{year}_{day}.pkl').format(
                    tile=tile, band=band, year=year, day=idx_row))):
            indexes = (np.isnan(tmp)) & (myd_mask == False)
            tmp[indexes] = predicted_aqua[indexes]
            np.save('intersection3.npy', im_subs)
            im_subs[indexes] = 3  # aqua id is "3" the final product
            sza[indexes] = myd_sza[indexes]
            saa[indexes] = myd_saa[indexes]
            vza[indexes] = myd_vza[indexes]
            vaa[indexes] = myd_vaa[indexes]

        # Fill second sensor
        # If the second score was from Terra then
        if (rank.loc[0][1] == "Terra") & (os.path.exists(os.path.join(outputModel,
                                                                      'SLSTR-Terra/random_forest_SLSTR-Terra_{tile}_{band}_{year}_{day}.pkl').format(
            tile=tile, band=band, year=year, day=idx_row))):
            indexes = (np.isnan(tmp)) & (mod_mask == False)
            tmp[indexes] = predicted_terra[indexes]
            np.save('intersection2.npy', im_subs)
            im_subs[indexes] = 2  # terra id is "2" the final product

            sza[indexes] = mod_sza[indexes]
            saa[indexes] = mod_saa[indexes]
            vza[indexes] = mod_vza[indexes]
            vaa[indexes] = mod_vaa[indexes]
        # If the first score was from Aqua then
        if (rank.loc[0][1] == "Aqua") & (os.path.exists(
                os.path.join(outputModel, 'SLSTR-Aqua/random_forest_SLSTR-Aqua_{tile}_{band}_{year}_{day}.pkl').format(
                    tile=tile, band=band, year=year, day=idx_row))):
            indexes = (np.isnan(tmp)) & (myd_mask == False)
            tmp[indexes] = predicted_aqua[indexes]
            np.save('intersection3.npy', im_subs)
            im_subs[indexes] = 3  # aqua id is "3" the final product
            sza[indexes] = myd_sza[indexes]
            saa[indexes] = myd_saa[indexes]
            vza[indexes] = myd_vza[indexes]
            vaa[indexes] = myd_vaa[indexes]

    else: # if there is no slstr info
        # fill terra
        mod_refl[mod_mask] = np.nan
        tmp = mod_refl.copy()
        im_subs = np.zeros(mod_mask.shape, dtype=np.uint8)

        im_subs[~np.isnan(mod_refl)] = 2  # terra id is "2" the final product
        sza[~np.isnan(mod_refl)] = mod_sza[~np.isnan(mod_refl)]
        saa[~np.isnan(mod_refl)] = mod_saa[~np.isnan(mod_refl)]
        vza[~np.isnan(mod_refl)] = mod_vza[~np.isnan(mod_refl)]
        vaa[~np.isnan(mod_refl)] = mod_vaa[~np.isnan(mod_refl)]

        # fill aqua
        # load model
        filename = os.path.join(outputModel,
                                'Terra-Aqua/random_forest_Terra-Aqua_{tile}_{band}_{year}_{day}.pkl'.format(
                                    tile=tile, band=band, year=year, day=idx_row))
        aqua_model = pickle.load(open(filename, 'rb'))
        # prepare predicted data
        d = pd.DataFrame()  # aqua
        d['land_cover'] = landcover.reshape(-1)  # invalid is 0
        d['dem'] = dem.reshape(-1)  # invalid is -214748
        d['slope'] = slope.reshape(-1)  # invalid is -9999
        d['aspect'] = aspect.reshape(-1)  # invalid -9999
        d['hillshade'] = hillshade.reshape(-1)  # invalid 0
        d['vza'] = myd_vza.reshape(-1)
        d['vaa'] = myd_vaa.reshape(-1)
        d['sza'] = myd_sza.reshape(-1)
        d['saa'] = myd_saa.reshape(-1)
        d['aqua'] = myd_refl.reshape(-1)
        d = d.fillna(0)
        d = np.array(d)
        predicted_aqua = aqua_model.predict(d)
        predicted_aqua = predicted_aqua.reshape(-1, 1)
        predicted_aqua = predicted_aqua.reshape(3600, 3600)

        indexes = (np.isnan(mod_refl)) & (myd_mask == False) & (watermask == True)
        tmp[indexes] = predicted_aqua[indexes]

        im_subs[indexes] = 3

        sza[indexes] = myd_sza[indexes]
        saa[indexes] = myd_saa[indexes]
        vza[indexes] = myd_vza[indexes]
        vaa[indexes] = myd_vaa[indexes]

    # save the image , mask and the angles
    name = outputModel + 'images/merged_random_forest_{tile}_{band}_{year}_{day}.npz'.format(tile=tile, band='red',
                                                                                             year=year, day=idx_row)
    # save variables as a compressed package
    np.savez(outputModel + 'images/merged_random_forest_{tile}_{band}_{year}_{day}.npz'.format(tile=tile, band='red',
                                                                                               year=year, day=idx_row),
             sza=sza, saa=saa, vza=vza, vaa=vaa, refl=tmp, mask=im_subs)
    print("finished from filling image")
