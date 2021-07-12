""" Run this script giving as argument the day of the year of interest"""
# packages:
import os
import netCDF4
import numpy as np
import glob

# import functions from other Scripts:
from Dem import *
from MathTimeFunctions import *
from PreprocessingFunctions import *
from LayerFunctions import *
from RandomForestModelsFunctions import *
from BRDFFunctions import *

####################################################################
''' - - - - - - - - - input parameter path - - - - - - - - - - -'''
####################################################################
MY_PATH = "/home/andrea/Desktop/BA_R/" # root path directory

MODIS_PATH = MY_PATH + 'MODIS/{product}/{year}/{tile}/'  # Where to put modis tile images
# Place raster modis for the projection, resampling ....
MODIS_RASTER = MY_PATH + 'MODIS/raster/'

MODIS_REPROJECTION = MY_PATH + 'MODIS/MODIS_reprojected/{tile}/{layer}/'

MODIS_REPROJECTED = MY_PATH + 'MODIS/MODIS_reprojected/h19v07/'

OUTPUT_PATH_MERGED = MY_PATH + 'integrated_images/2019/random_forest/h19v07/'

S3_PATH = MY_PATH + 'SYN/h19v07/'
S3_FILE = S3_PATH + 'SY_2_SYN-L3-P1D-h19v07-20190101-1.7.nc'
DemPath = MY_PATH + 'DEM/h19v07/'  # DEM

#####################################################################
''' - - - - - - - - - - - Algorithm Parameters - - - - - - - - - -'''
#####################################################################
dem_name = '/home/andrea/Desktop/BA_R/t/DEM/h19v07/dem.tif'
landcover_name = 'land_cover_300m-h19v07.tif'
tile = 'h19v07'
year = 2019
band = 'red'
dst_crs = 'EPSG:4326'
resampling_method = 'nearest'
# image_tile is a csv file with year, day of the year and the name of modis images for the tiles corresponding to syn tile
# Both MOD09GQ and MOD09GA, MYD09QG and MYD09GA should be in the csv
# Example of the header
# year  doy MOD09GQ_h18v07  MOD09GA_h18v07  MYD09GQ_h18v07  MYD09GA_h18v07  MOD09GQ_h19v07  MOD09GA_h19v07  MYD09GQ_h19v07  MYD09GA_h19v07
# with h18v08 , h19v07 corresponding to modis tiles that cover the same geographical area as syn tile.
image_tile = '/home/andrea/Desktop/BA_R/MODIS/images_tiles_h19v07.csv'
window = 14

#####################################################################
''' - - - - - - - - - - - - - Algorithm  - - - - - - - - - - - - -'''
#####################################################################



output_shape = (3600, 3600)

# from dem.tiff obtain slope.tiff, aspect.tiff and hillshade.tiff
slope = GenerateAttribute(dem_name, DemPath, attribute='slope')
aspect = GenerateAttribute(dem_name, DemPath, attribute='aspect')
hillshade = GenerateAttribute(dem_name, DemPath, attribute='hillshade')
landcover = get_layer_tiff(os.path.join(DemPath, landcover_name))
print('Attributes done')
water = np.where(landcover == 210, 1, 0)
NODATA = np.nan

""" ModisProcessing: Load Modis sensors (MOD09GA and MYD09GA) bands data of interest. "doy" is the day of the year"""
def ModisProcessing(doy):
    print("Start preprocessing MODIS sensors")
    process_nir_gq(2019, doy, sensor='MOD', layer="sur_refl_b06_1", img_tile=image_tile, input_path=MODIS_PATH,
                   modis_raster=MODIS_RASTER, output_path=MODIS_REPROJECTION, s3_file=S3_FILE)
    process_nir_gq(2019, doy, sensor='MYD', layer="sur_refl_b06_1", img_tile=image_tile, input_path=MODIS_PATH,
                   modis_raster=MODIS_RASTER, output_path=MODIS_REPROJECTION, s3_file=S3_FILE)
    process_nir_500m(2019, doy, sensor='MOD', layer='QC_500m_1', img_tile=image_tile, input_path=MODIS_PATH,
                     modis_raster=MODIS_RASTER, output_path=MODIS_REPROJECTION, s3_file=S3_FILE)
    process_nir_500m(2019, doy, sensor='MYD', layer='QC_500m_1', img_tile=image_tile, input_path=MODIS_PATH,
                     modis_raster=MODIS_RASTER, output_path=MODIS_REPROJECTION, s3_file=S3_FILE)
    process_nir_1km(2019, doy, sensor='MOD', layer='state_1km_1', img_tile=image_tile, input_path=MODIS_PATH,
                    modis_raster=MODIS_RASTER, output_path=MODIS_REPROJECTION, s3_file=S3_FILE)
    process_nir_1km(2019, doy, sensor='MYD', layer='state_1km_1', img_tile=image_tile, input_path=MODIS_PATH,
                    modis_raster=MODIS_RASTER, output_path=MODIS_REPROJECTION, s3_file=S3_FILE)
    process_nir_1km(2019, doy, sensor='MOD', layer='SensorZenith_1', img_tile=image_tile, input_path=MODIS_PATH,
                    modis_raster=MODIS_RASTER, output_path=MODIS_REPROJECTION, s3_file=S3_FILE)
    process_nir_1km(2019, doy, sensor='MOD', layer='SensorAzimuth_1', img_tile=image_tile, input_path=MODIS_PATH,
                    modis_raster=MODIS_RASTER, output_path=MODIS_REPROJECTION, s3_file=S3_FILE)
    process_nir_1km(2019, doy, sensor='MOD', layer='SolarZenith_1', img_tile=image_tile, input_path=MODIS_PATH,
                    modis_raster=MODIS_RASTER, output_path=MODIS_REPROJECTION, s3_file=S3_FILE)
    process_nir_1km(2019, doy, sensor='MOD', layer='SolarAzimuth_1', img_tile=image_tile, input_path=MODIS_PATH,
                    modis_raster=MODIS_RASTER, output_path=MODIS_REPROJECTION, s3_file=S3_FILE)
    process_nir_1km(2019, doy, sensor='MYD', layer='SensorZenith_1', img_tile=image_tile, input_path=MODIS_PATH,
                    modis_raster=MODIS_RASTER, output_path=MODIS_REPROJECTION, s3_file=S3_FILE)
    process_nir_1km(2019, doy, sensor='MYD', layer='SensorAzimuth_1', img_tile=image_tile, input_path=MODIS_PATH,
                    modis_raster=MODIS_RASTER, output_path=MODIS_REPROJECTION, s3_file=S3_FILE)
    process_nir_1km(2019, doy, sensor='MYD', layer='SolarZenith_1', img_tile=image_tile, input_path=MODIS_PATH,
                    modis_raster=MODIS_RASTER, output_path=MODIS_REPROJECTION, s3_file=S3_FILE)
    process_nir_1km(2019, doy, sensor='MYD', layer='SolarAzimuth_1', img_tile=image_tile, input_path=MODIS_PATH,
                    modis_raster=MODIS_RASTER, output_path=MODIS_REPROJECTION, s3_file=S3_FILE)
    print("Finish preprocessing MODIS sensors")
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        # Preprocess MODIS sensor bands
        ModisProcessing(int(sys.argv[1]))
        # Process random forest, create models using SLSTR as Reference for Terra and Aqua
        random_forest_models(year, int(sys.argv[1]), tile=tile, s3_path=S3_PATH, modis_reprojected=MODIS_REPROJECTED,
                             outputModel=OUTPUT_PATH_MERGED, DemPath=DemPath, LC=landcover_name)
        print('finish training the model')
        # Create composite image and save as a zip file
        random_forest(year, int(sys.argv[1]), tile=tile, s3_path=S3_PATH, modis_reprojected=MODIS_REPROJECTED,
                      outputModel=OUTPUT_PATH_MERGED, DemPath=DemPath, LC=landcover_name)
        print('finish generating the image')

        # To process brdf we need 1 week before and one week after, check in merged image exist, if not create it.
        # take the window and check for every day the existance of the image.  | int(sys.argv[1])>358
        if int(sys.argv[1]) < 8 or int(sys.argv[1]) > 358:
            print("BRDF can't be processed for the first nor the last week of the year")
        else:
            init = int(sys.argv[1]) - window // 2
            end = int(sys.argv[1]) + window // 2
            for i in range(init, end + 1):
                if not os.path.exists(
                        OUTPUT_PATH_MERGED + 'images/merged_random_forest_{tile}_{band}_{year}_{day}.npz'.format(
                                tile=tile, band=band, year=year, day=i)):
                    ModisProcessing(i)
                    random_forest_models(year, i, tile=tile, s3_path=S3_PATH, modis_reprojected=MODIS_REPROJECTED,
                                         outputModel=OUTPUT_PATH_MERGED, DemPath=DemPath, LC=landcover_name)
                    random_forest(year, i, tile=tile, s3_path=S3_PATH, modis_reprojected=MODIS_REPROJECTED,
                                  outputModel=OUTPUT_PATH_MERGED, DemPath=DemPath, LC=landcover_name)

            brdf_approach(i=int(sys.argv[1]), window=window, location=OUTPUT_PATH_MERGED, water=water, tile=tile,
                          year=year)
            print('finish brdf normalization')
            # save images in netCDF4 format
            f = netCDF4.Dataset(S3_FILE)
            # save netCDF4 file for brdf
            fn = OUTPUT_PATH_MERGED + 'brdf/brdf_merged_random_forest_{tile}_{band}_{year}_{day}.nc'.format(tile=tile,
                                                                                                            band=band,
                                                                                                            year=year,
                                                                                                            day=int(
                                                                                                                sys.argv[
                                                                                                                    1]))
            ds = netCDF4.Dataset(fn, 'w', format='NETCDF4')
            time = ds.createDimension('time', None)
            lat = ds.createDimension('lat', 3600)
            lon = ds.createDimension('lon', 3600)

            times = ds.createVariable('time', 'f4', ('time',))
            lats = ds.createVariable('lat', 'f4', ('lat',))
            lons = ds.createVariable('lon', 'f4', ('lon',))

            # fill
            lats[:] = f.variables['lat'][:].data  # get from syn
            lons[:] = f.variables['lon'][:].data  # get from syn

            # Variables
            refl = ds.createVariable('Refl', 'f4', ('time', 'lat', 'lon',))
            error = ds.createVariable('Error', 'f4', ('time', 'lat', 'lon',))
            refl.long_name = 'Surface Reflectance'
            dsa = np.load(
                OUTPUT_PATH_MERGED + 'brdf/brdf_merged_random_forest_{tile}_{band}_{year}_{day}.npz'.format(tile=tile,
                                                                                                            band=band,
                                                                                                            year=year,
                                                                                                            day=int(
                                                                                                                sys.argv[
                                                                                                                    1])))

            refl[0, :, :] = dsa['refl']
            error[0, :, :] = dsa['rmse']
            ds.close()

        f = netCDF4.Dataset(S3_FILE)
        fn = OUTPUT_PATH_MERGED + 'images/merged_random_forest_{tile}_{band}_{year}_{day}.nc'.format(tile=tile,
                                                                                                     band=band,
                                                                                                     year=year, day=int(
                sys.argv[1]))

        ds = netCDF4.Dataset(fn, 'w', format='NETCDF4')
        time = ds.createDimension('time', None)
        lat = ds.createDimension('lat', 3600)
        lon = ds.createDimension('lon', 3600)

        times = ds.createVariable('time', 'f4', ('time',))
        lats = ds.createVariable('lat', 'f4', ('lat',))
        lons = ds.createVariable('lon', 'f4', ('lon',))

        # fill
        lats[:] = f.variables['lat'][:]  # .data # get from syn
        lons[:] = f.variables['lon'][:]  # .data # get from syn
        # Variables
        refl = ds.createVariable('Refl', 'f4', ('time', 'lat', 'lon',))
        mask = ds.createVariable('Mask', 'f4', ('time', 'lat', 'lon',))
        vaa = ds.createVariable('vaa', 'f4', ('time', 'lat', 'lon',))
        vza = ds.createVariable('vza', 'f4', ('time', 'lat', 'lon',))
        saa = ds.createVariable('saa', 'f4', ('time', 'lat', 'lon',))
        sza = ds.createVariable('sza', 'f4', ('time', 'lat', 'lon',))

        refl.long_name = 'Surface Reflectance'
        # source of the images
        dsa = np.load(
            OUTPUT_PATH_MERGED + 'images/merged_random_forest_{tile}_{band}_{year}_{day}.npz'.format(tile=tile,
                                                                                                     band=band,
                                                                                                     year=year, day=int(
                    sys.argv[1])))

        refl[0, :, :] = dsa['refl']
        mask[0, :, :] = dsa['mask']

        vaa[0, :, :] = dsa['vaa']
        vza[0, :, :] = dsa['vza']
        saa[0, :, :] = dsa['saa']
        sza[0, :, :] = dsa['sza']
        ds.close()
        print("finished main")
