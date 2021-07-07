import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
import netCDF4 as nc4

from  MathTimeFunctions import *
from LayerFunctions import *

def random_forest_models(year, idx_row,tile, s3_path , modis_reprojected, outputModel, DemPath,LC, band = 'nir'):
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

    #Invalid data is : -9999
    aspect = get_layer_tiff(os.path.join(DemPath, 'aspect.tif'))
    #Invalid data is : -9999
    slope = get_layer_tiff(os.path.join(DemPath, 'slope.tif'))
    #Invalid data is : 0
    hillshade = get_layer_tiff(os.path.join(DemPath, 'hillshade.tif'))

    landcover = get_layer_tiff(os.path.join(DemPath, LC))
    #watermask = np.where(landcover == 210, 0, 1)

    error = pd.DataFrame(columns=['doy', 'Terra', 'Aqua', 'SLSTR'])
    error1 = pd.DataFrame(columns=['doy', 'Aqua'])

    date = doy2date(year, int(idx_row)).strftime("%Y%m%d")
    print(date)
    # load olci and terra
    s3_filename = 'SY_2_SYN-L3-P1D-{tile}-{date}-1.7.nc'.format(tile=tile, date=date)
    print(s3_filename)

    s3_ds = nc4.Dataset(os.path.join(s3_path, s3_filename))
    #s3_olci_refl = s3_ds['SDR_Oa17'][:] #Load OLCI data: Oa17 NIR Band, Wavelength(µm): 0.865
    #s3_slstr_refl = s3_ds['SDR_S3N'][:]
    s3_slstr_refl = s3_ds['SDR_S5N'][:] #Load SLSTR data: S5 SWIR Band , Wavelength(µm): 1.61

    error.loc[len(error)] = [np.nan, np.nan, np.nan, np.nan]

    s3_mask_olci = olci_mask(year, idx_row, s3_path, tile)
    s3_mask_slstr = slstr_mask(year, idx_row, s3_path, tile)
    # mask terra
    mod_refl = get_reflectance_layer_tiff(os.path.join(modis_reprojected, 'sur_refl_b06_1/MOD09GA_sur_refl_b06_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date))) #change
    print(os.path.join(modis_reprojected, 'sur_refl_b06_1/MOD09GA_sur_refl_b06_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    mod_mask = terra_mask(year, idx_row, modis_reprojected, tile)
    print("1:"+modis_reprojected)
    myd_refl = get_reflectance_layer_tiff(os.path.join(modis_reprojected, 'sur_refl_b06_1/MYD09GA_sur_refl_b06_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    myd_mask = aqua_mask(year, idx_row, modis_reprojected, tile)

    print("2:" + modis_reprojected)
    mod_vza = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SensorZenith_1/MOD09GA_SensorZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    mod_vaa = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SensorAzimuth_1/MOD09GA_SensorAzimuth_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    mod_sza = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SolarZenith_1/MOD09GA_SolarZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    mod_saa = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SolarAzimuth_1/MOD09GA_SolarAzimuth_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    
    myd_vza = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SensorZenith_1/MYD09GA_SensorZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    myd_vaa = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SensorAzimuth_1/MYD09GA_SensorAzimuth_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    myd_sza = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SolarZenith_1/MYD09GA_SolarZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    myd_saa = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SolarAzimuth_1/MYD09GA_SolarAzimuth_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))

    olci_vaa = s3_ds['OLC_VAA'][:]
    olci_vza = s3_ds['OLC_VZA'][:]
    s3_saa = s3_ds['SAA'][:]
    s3_sza = s3_ds['SZA'][:]
    #FIRST CONDITION: IF THERE IS DATA FOR THE REFERENCE THEN
    #if synergy exist and enough pixels 
    #if (s3_olci_refl.mask == False).sum() > 0: # OLCI as reference
    if (s3_slstr_refl.mask == False).sum() >0: # SLSTR as reference
        # olci-terra
        #calculate intersection, pixels with high quality in terra and olci
        intersection = np.zeros(mod_mask.shape, dtype=np.uint8)
        #intersection = (s3_mask_olci.data == True) & (mod_mask == False) # OLCI as reference: intersection
        intersection = (s3_mask_slstr.data == True) & (mod_mask == False)
        # fill the attributes
        df = pd.DataFrame()
        df['intersection'] = intersection.reshape(-1)
        df['land_cover'] = landcover.reshape(-1)
        df['dem'] = dem.reshape(-1) #invalid is -214748
        df['slope'] = slope.reshape(-1) #invalid is -9999
        df['aspect'] = aspect.reshape(-1) #invalid -9999
        df['hillshade'] = hillshade.reshape(-1) #invalid 0
        #terra sensors
        df['vza'] = mod_vza.reshape(-1)
        df['vaa'] = mod_vaa.reshape(-1)
        df['sza'] = mod_sza.reshape(-1)
        df['saa'] = mod_saa.reshape(-1)

        df['terra'] = mod_refl.reshape(-1)
        #df['olci'] = s3_olci_refl.reshape(-1) # OLCI as reference: put it as a column
        df['slstr'] = s3_slstr_refl.reshape(-1) # SLSTR as reference: put it as a column
        df = df[df.intersection == True]
        #delete invalid values
        df = df[df.dem > -214748]
        df = df[df.slope > -9999]
        df = df[df.aspect > -9999]
        df = df.drop('intersection', axis=1)
        l = df['land_cover'].value_counts(sort=True)
        #delete if there is a land cover with only one pixel
        for i in range(0, len(l)):
            if l.iloc[-1] < 2:
                print('clean '+str(list(l.keys())[-1]))
                df = df[df['land_cover'] != list(l.keys())[-1]]
                l = df['land_cover'].value_counts(sort=True)
            
        #prepare the data #save test and train set devide
        if len(df) > 1:
            #split test & train based on the land cover and save them
            df['label'] = df.index
            df.reset_index(drop=True, inplace=True)

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            for train_index, test_index in split.split(df, df['land_cover']):
                print(train_index, test_index)
                strat_train_set = df.loc[train_index]
                strat_test_set = df.loc[test_index]
            '''
            strat_test_set.to_csv(os.path.join(outputModel,'OLCI-Terra/test_set/testing_set_OLCI-Terra_{tile}_{band}_{year}_{day}.csv').format(
                                                                                            tile=tile, band=band, day=idx_row, year=year), index=False)

            strat_train_set.to_csv(os.path.join(outputModel,'OLCI-Terra/train_set/training_set_OLCI-Terra_{tile}_{band}_{year}_{day}.csv').format(
                                                                                            tile=tile, band=band, day=idx_row, year=year), index=False)
            #'''
            train = strat_train_set
            test = strat_test_set
            #run the model  and save it
            train = train.drop('label', axis=1)
            test = test.drop('label', axis=1)
            #y_train = np.array(train['olci']) #OLCI as reference
            #y_test = np.array(test['olci']) #OLCI as reference
            y_train = np.array(train['slstr']) # SLSTR as reference
            y_test = np.array(test['slstr']) # SLSTR as reference

            #drop the terra
            #train = train.drop('olci', axis=1) # OLCI as reference
            #test = test.drop('olci', axis=1) # OLCI as reference
            train = train.drop('slstr', axis=1) # SLSTR as reference
            test = test.drop('slstr', axis=1) # SLSTR as reference

            X_train = train
            X_test = test
            print('start terra model')
            #Random forest model
            regressor = RandomForestRegressor(max_depth=15, n_estimators=50, random_state=0)
            print(X_train)
            print(y_train)
            print(len(X_train))
            print(len(y_train))
            print(np.count_nonzero(np.isnan(X_train)))
            print(np.count_nonzero(np.isnan(y_train)))
            #lats = np.where(np.isnan(y_train))
            print("siguiente:")
            #y_train[lats]= 0
            regressor.fit(X_train, y_train)

            df.to_csv('data_Nan.csv', index=False, header=True)
            print("finish model")
            #save the models
            #pkl_name = os.path.join(outputModel,'OLCI-Terra/random_forest_OLCI-Terra_{tile}_{band}_{year}_{day}.pkl').format(
                               #tile=tile, band=band, day=idx_row, year=year)
            pkl_name = os.path.join(outputModel,
                                    'SLSTR-Terra/random_forest_SLSTR-Terra_{tile}_{band}_{year}_{day}.pkl').format(
                tile=tile, band=band, day=idx_row, year=year)
            with open(pkl_name, 'wb') as file:
                pickle.dump(regressor, file)

            y_pred = regressor.predict(test)
            error.loc[len(error)-1]['doy'] = idx_row
            print(np.count_nonzero(np.isnan(y_train)))
            print(np.count_nonzero(np.isnan(y_pred)))
            error.loc[len(error) -1]['Terra'] = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        #olci-aqua Model
        intersection = np.zeros(mod_mask.shape, dtype=np.uint8)
        intersection = (s3_mask_slstr.data == True) & (myd_mask == False)

        df = pd.DataFrame()
        df['intersection'] = intersection.reshape(-1)
        df['land_cover'] = landcover.reshape(-1)
        df['dem'] = dem.reshape(-1) #invalid is -214748
        df['slope'] = slope.reshape(-1) #invalid is -9999
        df['aspect'] = aspect.reshape(-1) #invalid -9999
        df['hillshade'] = hillshade.reshape(-1) #invalid 0
        #terra sensors
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
                print('clean '+str(list(l.keys())[-1]))
                df = df[df['land_cover'] != list(l.keys())[-1]]
                l = df['land_cover'].value_counts(sort=True)
        #prepare the data #save test and train set 
        if len(df) > 1:
            #split test & train and save them
            df['label'] = df.index
            df.reset_index(drop=True, inplace=True)

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            for train_index, test_index in split.split(df, df['land_cover']):
                strat_train_set = df.loc[train_index]
                strat_test_set = df.loc[test_index]
                '''
            strat_test_set.to_csv(os.path.join(outputModel,'OLCI-Aqua/test_set/testing_set_OLCI-Aqua_{tile}_{band}_{year}_{day}.csv').format(
                                                                                            tile=tile, band=band, day=idx_row, year=year), index=False)

            strat_train_set.to_csv(os.path.join(outputModel,'OLCI-Aqua/train_set/training_set_OLCI-Aqua_{tile}_{band}_{year}_{day}.csv').format(
                                                                                            tile=tile, band=band, day=idx_row, year=year), index=False)
            #'''
            train = strat_train_set
            test = strat_test_set
            #run the model  and save it
            train = train.drop('label', axis=1)
            test = test.drop('label', axis=1)
            #y_train = np.array(train['olci']) # OLCI as reference
            #y_test = np.array(test['olci']) # OLCI as reference
            y_train = np.array(train['slstr']) # SLSTR as reference
            y_test = np.array(test['slstr']) # SLSTR as reference

            #drop the terra
            #train = train.drop('olci', axis=1) # OLCI as reference
            #test = test.drop('olci', axis=1) # OLCI as reference
            train = train.drop('slstr', axis=1) # SLSTR as reference
            test = test.drop('slstr', axis=1) # SLSTR as reference

            X_train = train
            X_test = test
            print('start aqua model')

            print(np.count_nonzero(np.isnan(X_train)))
            print(np.count_nonzero(np.isnan(y_train)))
            regressor = RandomForestRegressor(max_depth=15, n_estimators=50, random_state=0)

            #lats = np.where(np.isnan(y_train))
            print("Aqua fit")
            #y_train[lats] = 0
            regressor.fit(X_train, y_train)

            print("finish model")

            #pkl_name = os.path.join(outputModel,'OLCI-Aqua/random_forest_OLCI-Aqua_{tile}_{band}_{year}_{day}.pkl').format(
                                #tile=tile, band=band, day=idx_row, year=year)
            pkl_name = os.path.join(outputModel,
                                    'SLSTR-Aqua/random_forest_SLSTR-Aqua_{tile}_{band}_{year}_{day}.pkl').format(
                tile=tile, band=band, day=idx_row, year=year)
            with open(pkl_name, 'wb') as file:
                pickle.dump(regressor, file)

            y_pred = regressor.predict(test)
            error.loc[len(error)-1]['doy'] = idx_row
            error.loc[len(error) -1]['Aqua'] = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

        """#olci-slstr 
        intersection = np.zeros(mod_mask.shape, dtype=np.uint8)
        intersection = (s3_mask_olci.data == True) & (s3_mask_slstr.data == True)

        df = pd.DataFrame()
        df['intersection'] = intersection.reshape(-1)
        df['land_cover'] = landcover.reshape(-1)
        df['dem'] = dem.reshape(-1) #invalid is -214748
        df['slope'] = slope.reshape(-1) #invalid is -9999
        df['aspect'] = aspect.reshape(-1) #invalid -9999
        df['hillshade'] = hillshade.reshape(-1) #invalid 0
        #terra sensors
        df['vza'] = olci_vza.reshape(-1)
        df['vaa'] = olci_vaa.reshape(-1)
        df['sza'] = s3_sza.reshape(-1)
        df['saa'] = s3_saa.reshape(-1)

        df['slstr'] = s3_slstr_refl.reshape(-1)
        df['olci'] = s3_olci_refl.reshape(-1)
        df = df[df.intersection == True]
        df = df[df.dem > -214748]
        df = df[df.slope > -9999]
        df = df[df.aspect > -9999]
        df = df.drop('intersection', axis=1)
        l = df['land_cover'].value_counts(sort=True)

        for i in range(0, len(l)):
            if l.iloc[-1] < 2:
                print('clean '+str(list(l.keys())[-1]))
                df = df[df['land_cover'] != list(l.keys())[-1]]
                l = df['land_cover'].value_counts(sort=True)
        #prepare the data #save test and train set devide
        if len(df) > 1:
            #split test & train and save them
            df['label'] = df.index
            df.reset_index(drop=True, inplace=True)

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            for train_index, test_index in split.split(df, df['land_cover']):
                strat_train_set = df.loc[train_index]
                strat_test_set = df.loc[test_index]
                '''
            strat_test_set.to_csv(os.path.join(outputModel,'OLCI-SLSTR/test_set/testing_set_OLCI-SLSTR_{tile}_{band}_{year}_{day}.csv').format(
                                                                                            tile=tile, band=band, day=idx_row, year=year), index=False)

            strat_train_set.to_csv(os.path.join(outputModel,'OLCI-SLSTR/train_set/training_set_OLCI-SLSTR_{tile}_{band}_{year}_{day}.csv').format(
                                                                                            tile=tile, band=band, day=idx_row, year=year), index=False)
            #'''
            train = strat_train_set
            test = strat_test_set
            #run the model  and save it
            train = train.drop('label', axis=1)
            test = test.drop('label', axis=1)
            y_train = np.array(train['olci'])
            y_test = np.array(test['olci'])

        #drop the terra
            train = train.drop('olci', axis=1)
            test = test.drop('olci', axis=1)

            X_train = train
            X_test = test
            print('start slstr model')

            regressor = RandomForestRegressor(max_depth=15, n_estimators=50, random_state=0)
            regressor.fit(X_train, y_train)
            print("finish model")

            pkl_name = os.path.join(outputModel,'OLCI-SLSTR/random_forest_OLCI-SLSTR_{tile}_{band}_{year}_{day}.pkl').format(
                              tile=tile, band=band, day=idx_row, year=year)
            with open(pkl_name, 'wb') as file:
                pickle.dump(regressor, file)

            y_pred = regressor.predict(test)
            error.loc[len(error)-1]['doy'] = idx_row
            error.loc[len(error) -1]['SLSTR'] = np.sqrt(metrics.mean_squared_error(y_test, y_pred))"""

        #No syn data or not enough
    else:
        # Terra is the reference and aqua
        # Train model between terra and Aqua
        
        intersection = np.zeros(mod_mask.shape, dtype=np.uint8)
        intersection = (myd_mask == False) & (mod_mask == False)

        df = pd.DataFrame()
        df['intersection'] = intersection.reshape(-1)
        df['land_cover'] = landcover.reshape(-1)
        df['dem'] = dem.reshape(-1) #invalid is -214748
        df['slope'] = slope.reshape(-1) #invalid is -9999
        df['aspect'] = aspect.reshape(-1) #invalid -9999
        df['hillshade'] = hillshade.reshape(-1) #invalid 0
        #terra sensors
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
        df = df.drop('intersection', axis = 1)
        l = df['land_cover'].value_counts(sort=True)

        for i in range(0, len(l)):
            if l.iloc[-1] <2:
                print('clean '+str(list(l.keys())[-1]))
                df = df[df['land_cover'] != list(l.keys())[-1]]
                l = df['land_cover'].value_counts(sort=True)
            
        #prepare the data #save test and train set devide
        if len(df) > 1:
            #split test & train and save them
            df['label'] = df.index
            df.reset_index(drop=True, inplace=True)

            split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
            for train_index, test_index in split.split(df, df['land_cover']):
                strat_train_set = df.loc[train_index]
                strat_test_set = df.loc[test_index]
                '''
            strat_test_set.to_csv(os.path.join(outputModel,'Terra-Aqua/test_set/testing_set_Terra-Aqua_{tile}_{band}_{year}_{day}.csv').format(
                                                                                            tile=tile, band=band, day=idx_row, year=year), index=False)

            strat_train_set.to_csv(os.path.join(outputModel,'Terra-Aqua/train_set/training_set_Terra-Aqua_{tile}_{band}_{year}_{day}.csv').format(
                                                                                            tile=tile, band=band, day=idx_row, year=year), index=False)
            #'''
            train = strat_train_set
            test = strat_test_set
            #run the model  and save it
            train = train.drop('label', axis=1)
            test = test.drop('label', axis=1)
            y_train = np.array(train['terra'])
            y_test = np.array(test['terra'])

        #drop the terra
            train = train.drop('terra', axis=1)
            test = test.drop('terra', axis=1)

            X_train = train
            X_test = test
            print('start terra aqua model')

            regressor = RandomForestRegressor(max_depth=15, n_estimators=50, random_state=0)
            regressor.fit(X_train, y_train)
            print("finish model")

            pkl_name = os.path.join(outputModel,'Terra-Aqua/random_forest_Terra-Aqua_{tile}_{band}_{year}_{day}.pkl').format(
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
        l = np.array(error[i:i+1]).reshape(4, -1)
        dictionary = array_to_dict(key, l)
        dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}
        #print(dictionary)
        for ke in dictionary:
            if np.isnan(dictionary[ke][0]):
                t[j] = np.nan
                j = j+1
            else:
                t[j] = ke
                j = j+1
        rank.loc[len(rank)] = t

    error.to_csv(os.path.join(outputModel, 'error_sensor_random_forest.csv'), index=False)
    error1.to_csv(os.path.join(outputModel, 'error_aqua.csv'), index=False)
    rank.to_csv(os.path.join(outputModel, 'ranking_sensors_random_forest.csv'), index=False)


def random_forest(year, idx_row,tile , s3_path , modis_reprojected, outputModel, DemPath, LC,band ='nir'):
    '''
    The function generate joined image
    Parameters
        idx_row : day of the year
        s3_file : path of synergy files
        modis_reprojected : location of the modis reprojected images
        outputModel : location of the output image
    '''
    dem = get_layer_tiff(os.path.join(DemPath, 'dem.tif'))

    #Invalid data is : -9999
    aspect = get_layer_tiff(os.path.join(DemPath, 'aspect.tif'))

        #Invalid data is : -9999

    slope = get_layer_tiff(os.path.join(DemPath, 'slope.tif'))

        #Invalid data is : 0
    hillshade = get_layer_tiff(os.path.join(DemPath, 'hillshade.tif'))
    landcover = get_layer_tiff(os.path.join(DemPath, LC))
    watermask = np.where(landcover == 210, 0, 1)
    idx_row = int(idx_row)
    # load the refl and angles
    rank = pd.read_csv(outputModel+'ranking_sensors_random_forest.csv')
    date = doy2date(year, int(idx_row)).strftime("%Y%m%d")
    
    s3_filename = 'SY_2_SYN-L3-P1D-{tile}-{date}-1.7.nc'.format(tile=tile, date=date)
    s3_ds = nc4.Dataset(os.path.join(s3_path, s3_filename))
    #s3_olci_refl = s3_ds['SDR_Oa17'][:] #OLCI as Reference
    #s3_slstr_refl = s3_ds['SDR_S3N'][:] #OLCI as Reference
    s3_slstr_refl = s3_ds['SDR_S5N'][:] # SLSTR as Reference
    #s3_olci_mask = olci_mask(year, idx_row, s3_path, tile)
    #Get Terra, Aqua & Mask
    mod_refl = get_reflectance_layer_tiff(os.path.join(modis_reprojected, 'sur_refl_b06_1/MOD09GA_sur_refl_b06_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    mod_mask = terra_mask(year, idx_row, modis_reprojected, tile)
    myd_refl = get_reflectance_layer_tiff(os.path.join(modis_reprojected, 'sur_refl_b06_1/MYD09GA_sur_refl_b06_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    myd_mask = aqua_mask(year, idx_row, modis_reprojected, tile)
    #angles
    mod_vza = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SensorZenith_1/MOD09GA_SensorZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    mod_vaa = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SensorAzimuth_1/MOD09GA_SensorAzimuth_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    mod_sza = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SolarZenith_1/MOD09GA_SolarZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    mod_saa = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SolarAzimuth_1/MOD09GA_SolarAzimuth_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    
    myd_vza = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SensorZenith_1/MYD09GA_SensorZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    myd_vaa = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SensorAzimuth_1/MYD09GA_SensorAzimuth_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    myd_sza = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SolarZenith_1/MYD09GA_SolarZenith_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))
    myd_saa = get_angle_layer_tiff(os.path.join(modis_reprojected, 'SolarAzimuth_1/MYD09GA_SolarAzimuth_1_{tile}_{date}_nearest_wgs84.tif'.format(tile=tile, date=date)))

    #olci_vaa = s3_ds['OLC_VAA'][:] #OLCI as Reference
    #olci_vza = s3_ds['OLC_VZA'][:] #OLCI as Reference
    s3_saa = s3_ds['SAA'][:]
    s3_sza = s3_ds['SZA'][:]

    """sza = np.zeros(s3_olci_refl.shape) # OLCI as Reference
    saa = np.zeros(s3_olci_refl.shape)
    vza = np.zeros(s3_olci_refl.shape)
    vaa = np.zeros(s3_olci_refl.shape)"""

    sza = np.zeros(s3_slstr_refl.shape)
    saa = np.zeros(s3_slstr_refl.shape)
    vza = np.zeros(s3_slstr_refl.shape)
    vaa = np.zeros(s3_slstr_refl.shape)

    #if synergy exist, then it's the refrence

    #if (s3_olci_refl.mask == False).sum() > 0: # OLCI as Reference
    if (s3_slstr_refl.mask == False).sum() > 0: # SLSTR as Reference
        #fill with reference
        tmp = s3_slstr_refl.copy()
        im_subs = np.zeros(s3_slstr_refl.shape, dtype=np.uint8)
        im_subs[~np.isnan(s3_slstr_refl)] = 1  # OLCI take value 1 in pseudocode
        #put angles
        sza[~np.isnan(s3_slstr_refl)] = s3_sza[~np.isnan(s3_slstr_refl)] # SLSTR as Reference
        saa[~np.isnan(s3_slstr_refl)] = s3_saa[~np.isnan(s3_slstr_refl)] # SLSTR as Reference
        #vza[~np.isnan(s3_olci_refl)] = olci_vza[~np.isnan(s3_olci_refl)]
        #vaa[~np.isnan(s3_olci_refl)] = olci_vaa[~np.isnan(s3_olci_refl)]

        # load regressor model
        #filename = os.path.join(outputModel , 'OLCI-Terra/random_forest_OLCI-Terra_{tile}_{band}_{year}_{day}.pkl'.format(
                                                #tile=tile, band=band, year=year, day=idx_row)) # OLCI as Reference
        filename = os.path.join(outputModel,
                                'SLSTR-Terra/random_forest_SLSTR-Terra_{tile}_{band}_{year}_{day}.pkl'.format(
            tile = tile, band = band, year = year, day = idx_row)) # SLSTR as Reference
        if os.path.exists(filename):
            print("HA ENTRADO POR IF SLSTR-Terra")
            terra_model = pickle.load(open(filename, 'rb'))
            d = pd.DataFrame()  # terra
            d['land_cover'] = landcover.reshape(-1) # invalid is 0
            d['dem'] = dem.reshape(-1) # invalid is -214748
            d['slope'] = slope.reshape(-1) #invalid is -9999
            d['aspect'] = aspect.reshape(-1) #invalid -9999
            d['hillshade'] = hillshade.reshape(-1) #invalid 0
            d['vza'] = mod_vza.reshape(-1)
            d['vaa'] = mod_vaa.reshape(-1)
            d['sza'] = mod_sza.reshape(-1)
            d['saa'] = mod_saa.reshape(-1)
            d['terra'] = mod_refl.reshape(-1)
            d = d.fillna(0)
            d = np.array(d)
            predicted_terra = terra_model.predict(d)
            predicted_terra = predicted_terra.reshape(-1, 1)
            predicted_terra = predicted_terra.reshape(3600, 3600)




        # Load OLCI-Aqua model
        #filename = os.path.join(outputModel, 'OLCI-Aqua/random_forest_OLCI-Aqua_{tile}_{band}_{year}_{day}.pkl'.format(
                                                #tile=tile, band=band, year=year, day=idx_row)) # OLCI as Reference
        filename = os.path.join(outputModel, 'SLSTR-Aqua/random_forest_SLSTR-Aqua_{tile}_{band}_{year}_{day}.pkl'.format(
            tile=tile, band=band, year=year, day=idx_row)) # SLSTR as Reference

        if os.path.exists(filename):

            print("HA ENTRADO POR IF OLCI-Aqua")
            aqua_model = pickle.load(open(filename, 'rb'))

            d = pd.DataFrame()  # aqua
            d['land_cover'] = landcover.reshape(-1) # invalid is 0
            d['dem'] = dem.reshape(-1) # invalid is -214748
            d['slope'] = slope.reshape(-1) #invalid is -9999
            d['aspect'] = aspect.reshape(-1) #invalid -9999
            d['hillshade'] = hillshade.reshape(-1) #invalid 0
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


        """# Load OLCI-SLSTR
        filename = os.path.join(outputModel, 'OLCI-SLSTR/random_forest_OLCI-SLSTR_{tile}_{band}_{year}_{day}.pkl'.format(
                                                tile=tile, band=band, year=year, day=idx_row))
        if os.path.exists(filename):

            print("HA ENTRADO POR IF OLCI-SLSTR")
            slstr_model = pickle.load(open(filename, 'rb'))

        #prepare the predicted images
            d = pd.DataFrame()  # slstr
            d['land_cover'] = landcover.reshape(-1) # invalid is 0
            d['dem'] = dem.reshape(-1) # invalid is -214748
            d['slope'] = slope.reshape(-1) #invalid is -9999
            d['aspect'] = aspect.reshape(-1) #invalid -9999
            d['hillshade'] = hillshade.reshape(-1) #invalid 0
            d['vza'] = olci_vza.reshape(-1)
            d['vaa'] = olci_vaa.reshape(-1)
            d['sza'] = s3_sza.reshape(-1)
            d['saa'] = s3_saa.reshape(-1)
            d['slstr'] = s3_slstr_refl.reshape(-1)
            d= d.fillna(0)
            d = np.array(d)
            predicted_slstr =  slstr_model.predict(d)
            predicted_slstr = predicted_slstr.reshape(-1, 1)
            predicted_slstr = predicted_slstr.reshape(3600, 3600)
        print('okk')"""
        #Fill first sensor pixels
        """if (rank.loc[0][0] == "SLSTR" ) & (os.path.exists(os.path.join(outputModel, 'OLCI-SLSTR/random_forest_OLCI-SLSTR_{tile}_{band}_{year}_{day}.pkl').format(tile=tile, band=band, year=year, day=idx_row))):

            indexes = (np.isnan(tmp)) & (~np.isnan(s3_slstr_refl))
            tmp[indexes] = predicted_slstr[indexes]
            im_subs[indexes] = 2

            #fill angles
            sza[indexes] = s3_sza[indexes]
            saa[indexes] = s3_saa[indexes]
            vza[indexes] = olci_vza[indexes]
            vaa[indexes] = olci_vaa[indexes]"""
        print("SLSTR[0][0]")
        if (rank.loc[0][0] == "Terra") & (os.path.exists(os.path.join(outputModel, 'SLSTR-Terra/random_forest_SLSTR-Terra_{tile}_{band}_{year}_{day}.pkl').format(tile=tile, band=band, year=year, day=idx_row))):
            indexes = (np.isnan(tmp)) & (mod_mask == False)
            tmp[indexes] = predicted_terra[indexes]
            im_subs[indexes] = 2

            sza[indexes] = mod_sza[indexes]
            saa[indexes] = mod_saa[indexes]
            vza[indexes] = mod_vza[indexes]
            vaa[indexes] = mod_vaa[indexes]
        print("Terra[0][0]")
        if (rank.loc[0][0] == "Aqua") & (os.path.exists(os.path.join(outputModel, 'SLSTR-Aqua/random_forest_SLSTR-Aqua_{tile}_{band}_{year}_{day}.pkl').format(tile=tile, band=band, year=year, day=idx_row))):

            indexes = (np.isnan(tmp)) & (myd_mask == False)
            tmp[indexes] = predicted_aqua[indexes]
            im_subs[indexes] = 3
            sza[indexes] = myd_sza[indexes]
            saa[indexes] = myd_saa[indexes]
            vza[indexes] = myd_vza[indexes]
            vaa[indexes] = myd_vaa[indexes]
        print("Aqua[0][0]")
        #fill second sensor
        """if (rank.loc[0][1] == "SLSTR")& (os.path.exists(os.path.join(outputModel, 'OLCI-SLSTR/random_forest_OLCI-SLSTR_{tile}_{band}_{year}_{day}.pkl').format(tile=tile, band=band, year=year, day=idx_row))):

            indexes = (np.isnan(tmp)) & (~np.isnan(s3_slstr_refl))
            tmp[indexes] = predicted_slstr[indexes]
            im_subs[indexes] = 2

            #fill angles
            sza[indexes] = s3_sza[indexes]
            saa[indexes] = s3_saa[indexes]
            vza[indexes] = olci_vza[indexes]
            vaa[indexes] = olci_vaa[indexes]
        print("SLSTR[0][1]")"""
        if (rank.loc[0][1] == "Terra" ) & (os.path.exists(os.path.join(outputModel , 'SLSTR-Terra/random_forest_SLSTR-Terra_{tile}_{band}_{year}_{day}.pkl').format(tile=tile, band=band, year=year, day=idx_row))):

            indexes = (np.isnan(tmp)) & (mod_mask == False)
            tmp[indexes] = predicted_terra[indexes]
            im_subs[indexes] = 2

            sza[indexes] = mod_sza[indexes]
            saa[indexes] = mod_saa[indexes]
            vza[indexes] = mod_vza[indexes]
            vaa[indexes] = mod_vaa[indexes]
        print("Terra[0][1]")
        if (rank.loc[0][1] == "Aqua") & (os.path.exists(os.path.join(outputModel , 'SLSTR-Aqua/random_forest_SLSTR-Aqua_{tile}_{band}_{year}_{day}.pkl').format(tile=tile, band=band, year=year, day=idx_row))):

            indexes = (np.isnan(tmp)) & (myd_mask == False)
            tmp[indexes] = predicted_aqua[indexes]
            im_subs[indexes] = 3
            sza[indexes] = myd_sza[indexes]
            saa[indexes] = myd_saa[indexes]
            vza[indexes] = myd_vza[indexes]
            vaa[indexes] = myd_vaa[indexes]
        print("Aqua[0][1]")
        #Fill the last pixel
        """if (rank.loc[0][2] == "SLSTR") & (os.path.exists(os.path.join(outputModel, 'OLCI-SLSTR/random_forest_OLCI-SLSTR_{tile}_{band}_{year}_{day}.pkl').format(tile=tile, band=band, year=year, day=idx_row))):

            indexes = (np.isnan(tmp)) & (~np.isnan(s3_slstr_refl))
            tmp[indexes] = predicted_slstr[indexes]
            im_subs[indexes] = 2

            #fill angles
            sza[indexes] = s3_sza[indexes]
            saa[indexes] = s3_saa[indexes]
            vza[indexes] = olci_vza[indexes]
            vaa[indexes] = olci_vaa[indexes]"""

        """if (rank.loc[0][2] == "Terra") & (os.path.exists(os.path.join(outputModel , 'OLCI-Terra/random_forest_OLCI-Terra_{tile}_{band}_{year}_{day}.pkl').format(tile=tile, band=band, year=year, day=idx_row))):

            indexes = (np.isnan(tmp)) & (mod_mask == False)
            tmp[indexes] = predicted_terra[indexes]
            im_subs[indexes] = 3

            sza[indexes] = mod_sza[indexes]
            saa[indexes] = mod_saa[indexes]
            vza[indexes] = mod_vza[indexes]
            vaa[indexes] = mod_vaa[indexes]

        if (rank.loc[0][2] == "Aqua" ) & (os.path.exists(os.path.join(outputModel, 'OLCI-Aqua/random_forest_OLCI-Aqua_{tile}_{band}_{year}_{day}.pkl').format(tile=tile, band=band, year=year, day=idx_row))):

            indexes = (np.isnan(tmp)) & (myd_mask == False)
            tmp[indexes] = predicted_aqua[indexes]
            im_subs[indexes] = 4
            sza[indexes] = myd_sza[indexes]
            saa[indexes] = myd_saa[indexes]
            vza[indexes] = myd_vza[indexes]
            vaa[indexes] = myd_vaa[indexes]"""
        # if there is not  synergy file
        print("Fin[0][2]")
    else:
        # fill terra
        mod_refl[mod_mask] = np.nan
        tmp = mod_refl.copy()

        im_subs = np.zeros(mod_mask.shape, dtype=np.uint8)
        im_subs[~np.isnan(mod_refl)] = 2 #3
        sza[~np.isnan(mod_refl)] = mod_sza[~np.isnan(mod_refl)]
        saa[~np.isnan(mod_refl)] = mod_saa[~np.isnan(mod_refl)]
        vza[~np.isnan(mod_refl)] = mod_vza[~np.isnan(mod_refl)]
        vaa[~np.isnan(mod_refl)] = mod_vaa[~np.isnan(mod_refl)]

        #fill aqua
        # load model
        filename = os.path.join(outputModel, 'Terra-Aqua/random_forest_Terra-Aqua_{tile}_{band}_{year}_{day}.pkl'.format(
                                                tile=tile, band=band, year=year, day=idx_row))
        aqua_model = pickle.load(open(filename, 'rb'))
        #prepare predicted data
        d = pd.DataFrame()  # aqua
        d['land_cover'] = landcover.reshape(-1) # invalid is 0
        d['dem'] = dem.reshape(-1) # invalid is -214748
        d['slope'] = slope.reshape(-1) #invalid is -9999
        d['aspect'] = aspect.reshape(-1) #invalid -9999
        d['hillshade'] = hillshade.reshape(-1) #invalid 0
        d['vza'] = myd_vza.reshape(-1)
        d['vaa'] = myd_vaa.reshape(-1)
        d['sza'] = myd_sza.reshape(-1)
        d['saa'] = myd_saa.reshape(-1)
        d['aqua'] = myd_refl.reshape(-1)
        d = d.fillna(0)
        d = np.array(d)
        predicted_aqua =  aqua_model.predict(d)
        predicted_aqua = predicted_aqua.reshape(-1, 1)
        predicted_aqua = predicted_aqua.reshape(3600, 3600)


        indexes = (np.isnan(mod_refl)) & (myd_mask == False) & (watermask == True)
        tmp[indexes] = predicted_aqua[indexes]

        im_subs[indexes] = 3 #4

        sza[indexes] = myd_sza[indexes]
        saa[indexes] = myd_saa[indexes]
        vza[indexes] = myd_vza[indexes]
        vaa[indexes] = myd_vaa[indexes]

    #save the image , mask and the angles
    print("fin de fin")
    name = outputModel+ 'images/merged_random_forest_{tile}_{band}_{year}_{day}.npz'.format(tile=tile, band='nir', year=year, day=idx_row)
    print(name)
    print(tile,year,idx_row)
    #save variables as a compressed package
    #merged_random_forest_{tile}_{band}_{year}_{day}.npz'.format(tile=tile, band='nir', year=year, day=idx_row)
    np.savez(outputModel+ 'images/merged_random_forest_{tile}_{band}_{year}_{day}.npz'.format(tile=tile, band='nir', year=year, day=idx_row), sza=sza, saa=saa, vza=vza, vaa=vaa, refl=tmp, mask=im_subs)
    #
    print("acaba...")