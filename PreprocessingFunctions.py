import os
import glob
import numpy as np
import pandas as pd
import netCDF4 as nc4
import rasterio.merge
import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds

from MathTimeFunctions import *

dst_crs = 'EPSG:4326'
def merge(input1, res=None, nodata=None, precision=4):
    import warnings
    warnings.warn("Deprecated; Use rasterio.merge instead", DeprecationWarning)
    return rasterio.merge.merge(input1, res, nodata, precision)


def process_nir_gq(year, doy, sensor, layer, img_tile, input_path, modis_raster, output_path, s3_file, resampling_method = 'nearest'):
    print("ENTRA -process_nir_gq-:")
    # create directories to store the images
    if not os.path.exists(modis_raster+'/tiles'):
        os.mkdir(modis_raster+'/tiles')
    if not os.path.exists(modis_raster+'/merged'):
        os.mkdir(modis_raster+'/merged')
    if not os.path.exists(modis_raster+'/reprojected'):
        os.mkdir(modis_raster+'/reprojected')
    # read .csv where there is the name of MODIS images.  With the following header
    # year, doy, product1_tile1, product_2_tile1, ….., productN_tileM
    # with productN is [MOD09GA,MOD09GQ,MYD09GA,MYD09GQ]
    # tileM corresponsing the the name of the tiles that cover syn tile
    df = pd.read_csv(img_tile)
    #take row with the year and day of interest
    df = df[(df.year == year) & (df.doy == doy)]
    #take only values corresponding to the "MOD09" sensor
    columns = [v for v in df.columns.values if (v not in ['year', 'doy']) & (sensor in v)]
    #take the MOD09GQ values
    print(columns)
    #gq_images = [v for v in columns if 'GQ' in v]
    # product GA : MODIS Terra Surface Reflectance Daily L2G Global 500 m and 1 km
    gq_images = [v for v in columns if 'GA' in v]

    print(gq_images)
    for gq in gq_images:
        #ga = gq.replace('GQ', 'GA')
        tile = gq.split('_')[1]
        product = gq.split('_')[0]

        gq_file = os.path.join(input_path.format(product=gq.split('_')[0], year=year, tile=tile), df[gq].values[0])
        #ga_file = os.path.join(DATA_PATH.format(product=ga.split('_')[0], year=year, tile=tile), df[ga].values[0])

        date = os.path.basename(gq_file).split('.')[1].replace('A', '')
        print(date)
        print(gq_file, layer)
        ######## CHANGES :Andrea ###########
        #with rio.open('HDF4_EOS:EOS_GRID:"{}":MODIS_Grid_2D:{}'.format(gq_file, layer)) as src:
        with rio.open('HDF4_EOS:EOS_GRID:"{}":MODIS_Grid_500m_2D:{}'.format(gq_file, layer)) as src:
            refl = np.squeeze(src.read())
            profile = src.profile
            profile.update(
                #dtype=rio.float32,
                count=1,
                compress='lzw'#,
                #nodata=NODATA
            )
            filename = '{product}_{tile}_{date}_{layer}.tif'.format(product=product, tile=tile, 
                                                                    date=date, layer=layer)
            print(os.path.join(modis_raster+'tiles/' , filename))
            with rio.open(os.path.join(modis_raster+'tiles/' , filename), 'w', **profile) as dst:
                dst.write(refl, 1)

    join_images = []
    for gq in gq_images:
        tile = gq.split('_')[1]
        product = gq.split('_')[0]
        date = os.path.basename(gq_file).split('.')[1].replace('A', '')
        filename = '{product}_{tile}_{date}_{layer}.tif'.format(product=product, tile=tile, 
                                                                date=date, layer=layer)
        join_images.append(rio.open(os.path.join(modis_raster+'tiles/', filename), 'r'))
        print('%%%%%%%%%%')
        print(os.path.join(modis_raster+'tiles/', filename))
    joint_image, output_transform = merge(join_images)
    out_meta = join_images[0].meta.copy()
    out_meta.update({'driver': 'GTiff',
                     'height': joint_image.shape[1],
                     'width': joint_image.shape[2],
                     'transform': output_transform})
    merged_file = '{product}_{date}_{layer}.tif'.format(product=product, date=date, layer=layer)

    with rasterio.open(os.path.join(modis_raster+'merged/', merged_file), "w", **out_meta) as dest:
        dest.write(joint_image)

    if resampling_method == 'cubic':
        resampling = Resampling.cubic
    elif resampling_method == 'bilinear':
        resampling = Resampling.bilinear
    elif resampling_method == 'lanczos':
        resampling = Resampling.lanczos
    elif resampling_method == 'nearest':
        resampling = Resampling.nearest

    ####
    date = doy2date(int(df.year.values[0]), int(df.doy.values[0])).strftime("%Y%m%d")

    tile_name = os.path.basename(s3_file).split('-')[3]
    print('%%%%%SYNERGY%%%%%%')
    print('# read SLSTR SWIR band:')
    #src_s3 = rasterio.open('NETCDF:"{}":SDR_Oa17'.format(s3_file)) # OLCI as Reference
    src_s3 = rasterio.open('NETCDF:"{}":SDR_S5N'.format(s3_file))  # SLSTR as Reference
    width, height = src_s3.width, src_s3.height


    with rasterio.open(os.path.join(modis_raster+'merged/', merged_file)) as src:
        transform, width, height = calculate_default_transform(
            src_crs=src.crs, dst_crs=dst_crs, width=src.width, height=src.height, left=src.bounds.left,
            bottom=src.bounds.bottom, right=src.bounds.right, top=src.bounds.top, resolution=src_s3.res)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height#,
            #'nodata': NODATA
        })

        with rasterio.open(os.path.join(modis_raster+'reprojected/', merged_file),
                           'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    #src_nodata=NODATA,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    #dst_nodata=NODATA,
                    resampling=resampling)

    with rasterio.open(os.path.join(modis_raster+'reprojected/', merged_file)) as src:
        window = from_bounds(top=src_s3.bounds.top, bottom=src_s3.bounds.bottom,
                             left=src_s3.bounds.left, right=src_s3.bounds.right,
                             transform=src.transform)
        out_image = src.read(1, window=window)
        #out_image[out_image>1.6] = NODATA
        #out_image[out_image<-0.01] = NODATA
        out_meta = src.meta.copy()
        out_meta.update({"width":src_s3.width, "height":src_s3.height, "transform":src_s3.transform})
        output_file = '{}_{}_{}_{}_{}_wgs84.tif'.format(product, layer, tile_name, date, resampling_method)
        print("REVISE PATH")
        if not os.path.exists(output_path.format(tile=tile_name, layer=layer)):
            os.mkdir(output_path.format(tile=tile_name, layer=layer))
        print(os.path.join(output_path.format(tile=tile_name, layer=layer), output_file))
        with rasterio.open(os.path.join(output_path.format(tile=tile_name, layer=layer), output_file), "w", **out_meta) as dest:
            dest.write(np.expand_dims(out_image, axis=0))



def process_nir_500m(year, doy, sensor, layer , img_tile, input_path, modis_raster, output_path , s3_file, resampling_method = 'nearest'):
    print("ENTRA -process_nir_500m-:")
    if not os.path.exists(modis_raster+'/tiles'):
        os.mkdir(modis_raster+'/tiles')
    if not os.path.exists(modis_raster+'/merged'):
        os.mkdir(modis_raster+'/merged')
    if not os.path.exists(modis_raster+'/reprojected'):
        os.mkdir(modis_raster+'/reprojected')
    df = pd.read_csv(img_tile)
    df = df[(df.year == year) & (df.doy == doy)]
    print("process_nir_500m")
    columns = [v for v in df.columns.values if (v not in ['year', 'doy']) & (sensor in v)]
    gq_images = [v for v in columns if 'GQ' in v]
    #gq_images = [v for v in columns if 'GA' in v]

    for gq in gq_images:
        ga = gq.replace('GQ', 'GA')

        tile = gq.split('_')[1]
        product = ga.split('_')[0]

        

        #gq_file = os.path.join(DATA_PATH.format(product=gq.split('_')[0], year=year, tile=tile), df[gq].values[0])
        ga_file = os.path.join(input_path.format(product=ga.split('_')[0], year=year, tile=tile), df[ga].values[0])


        date = os.path.basename(ga_file).split('.')[1].replace('A', '')

        #refl = preprocess_nir(gq_file, ga_file)
        print(ga_file, layer)
        with rio.open('HDF4_EOS:EOS_GRID:"{}":MODIS_Grid_500m_2D:{}'.format(ga_file, layer)) as src:
            refl = np.squeeze(src.read())
            profile = src.profile
            profile.update(
                #dtype=rio.float32,
                count=1,
                compress='lzw'#,
                #nodata=NODATA
            )
            filename = '{product}_{tile}_{date}_{layer}.tif'.format(product=product, tile=tile, 
                                                                    date=date, layer=layer)
            with rio.open(os.path.join(modis_raster+'tiles/', filename), 'w', **profile) as dst:
                dst.write(refl, 1)

    join_images = []
    for gq in gq_images:
        tile = gq.split('_')[1]
        product = ga.split('_')[0]
        date = os.path.basename(ga_file).split('.')[1].replace('A', '')
        filename = '{product}_{tile}_{date}_{layer}.tif'.format(product=product, tile=tile, 
                                                                date=date, layer=layer)

        join_images.append(rio.open(os.path.join(modis_raster+'tiles/', filename), 'r'))

    joint_image, output_transform=merge(join_images)
    out_meta = join_images[0].meta.copy()
    out_meta.update({'driver': 'GTiff',
                     'height': joint_image.shape[1],
                     'width': joint_image.shape[2],
                     'transform': output_transform})
    merged_file = '{product}_{date}_{layer}.tif'.format(product=product, date=date, layer=layer)

    with rasterio.open(os.path.join(modis_raster+'merged/', merged_file), "w", **out_meta) as dest:
        dest.write(joint_image)

    if resampling_method == 'cubic':
        resampling = Resampling.cubic
    elif resampling_method == 'bilinear':
        resampling = Resampling.bilinear
    elif resampling_method == 'lanczos':
        resampling = Resampling.lanczos
    elif resampling_method == 'nearest':
        resampling = Resampling.nearest

    ####
    date = doy2date(int(df.year.values[0]), int(df.doy.values[0])).strftime("%Y%m%d")

    tile_name = os.path.basename(s3_file).split('-')[3]

    src_s3 = rasterio.open('NETCDF:"{}":SDR_Oa17'.format(s3_file))
    width, height = src_s3.width, src_s3.height


    with rasterio.open(os.path.join(modis_raster+'merged/', merged_file)) as src:
        transform, width, height = calculate_default_transform(
            src_crs=src.crs, dst_crs=dst_crs, width=src.width, height=src.height, left=src.bounds.left,
            bottom=src.bounds.bottom, right=src.bounds.right, top=src.bounds.top, resolution=src_s3.res)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height#,
            #'nodata': NODATA
        })

        with rasterio.open(os.path.join(modis_raster+'reprojected/', merged_file),
                           'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    #src_nodata=NODATA,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    #dst_nodata=NODATA,
                    resampling=resampling)

    with rasterio.open(os.path.join(modis_raster+'reprojected/', merged_file)) as src:
        window = from_bounds(top=src_s3.bounds.top, bottom=src_s3.bounds.bottom,
                             left=src_s3.bounds.left, right=src_s3.bounds.right,
                             transform=src.transform)
        out_image = src.read(1, window=window)
        #out_image[out_image>1.6] = NODATA
        #out_image[out_image<-0.01] = NODATA
        out_meta = src.meta.copy()
        out_meta.update({"width":src_s3.width, "height":src_s3.height, "transform":src_s3.transform})

        output_file = '{}_{}_{}_{}_{}_wgs84.tif'.format(product, layer, tile_name, date, resampling_method)

        with rasterio.open(os.path.join(output_path.format(tile = tile_name , layer = layer ), output_file), "w", **out_meta) as dest:
            dest.write(np.expand_dims(out_image, axis=0))

def process_nir_1km(year, doy, sensor, layer , img_tile, input_path, modis_raster, output_path, s3_file, resampling_method = 'nearest'):
    print("ENTRA -process_nir_1km-:")
    if not os.path.exists(modis_raster+'/tiles'):
        os.mkdir(modis_raster+'/tiles')
    if not os.path.exists(modis_raster+'/merged'):
        os.mkdir(modis_raster+'/merged')
    if not os.path.exists(modis_raster+'/reprojected'):
        os.mkdir(modis_raster+'/reprojected')
    df = pd.read_csv(img_tile)
    df = df[(df.year == year) & (df.doy == doy)]
    print("process_nir_1km")
    columns = [v for v in df.columns.values if (v not in ['year', 'doy']) & (sensor in v)]
    ######### CHANGES: Andrea ########
    gq_images = [v for v in columns if 'GQ' in v]
    #gq_images = [v for v in columns if 'GA' in v]
    for gq in gq_images:
        ga = gq.replace('GQ', 'GA')

        tile = gq.split('_')[1]
        product = ga.split('_')[0]

        

        #gq_file = os.path.join(DATA_PATH.format(product=gq.split('_')[0], year=year, tile=tile), df[gq].values[0])
        ga_file = os.path.join(input_path.format(product=ga.split('_')[0], year=year, tile=tile), df[ga].values[0])

        date = os.path.basename(ga_file).split('.')[1].replace('A', '')

        #refl = preprocess_nir(gq_file, ga_file)
        print(ga_file, layer)
        with rio.open('HDF4_EOS:EOS_GRID:"{}":MODIS_Grid_1km_2D:{}'.format(ga_file, layer)) as src:
            refl = np.squeeze(src.read())
            profile = src.profile
            profile.update(
                #dtype=rio.float32,
                count=1,
                compress='lzw'#,
                #nodata=NODATA
            )
            filename = '{product}_{tile}_{date}_{layer}.tif'.format(product=product, tile=tile, 
                                                                    date=date, layer=layer)
            with rio.open(os.path.join(modis_raster+'tiles/', filename), 'w', **profile) as dst:
                dst.write(refl, 1)

    join_images = []
    for gq in gq_images:
        tile = gq.split('_')[1]
        product = ga.split('_')[0]
        date = os.path.basename(ga_file).split('.')[1].replace('A', '')
        filename = '{product}_{tile}_{date}_{layer}.tif'.format(product=product, tile=tile, 
                                                                date=date, layer=layer)

        join_images.append(rio.open(os.path.join(modis_raster+'tiles/', filename), 'r'))

    joint_image, output_transform=merge(join_images)
    out_meta = join_images[0].meta.copy()
    out_meta.update({'driver': 'GTiff',
                     'height': joint_image.shape[1],
                     'width': joint_image.shape[2],
                     'transform': output_transform})
    merged_file = '{product}_{date}_{layer}.tif'.format(product=product, date=date, layer=layer)

    with rasterio.open(os.path.join(modis_raster+'merged/', merged_file), "w", **out_meta) as dest:
        dest.write(joint_image)

    if resampling_method == 'cubic':
        resampling = Resampling.cubic
    elif resampling_method == 'bilinear':
        resampling = Resampling.bilinear
    elif resampling_method == 'lanczos':
        resampling = Resampling.lanczos
    elif resampling_method == 'nearest':
        resampling = Resampling.nearest

    ####
    date = doy2date(int(df.year.values[0]), int(df.doy.values[0])).strftime("%Y%m%d")

    tile_name = os.path.basename(s3_file).split('-')[3]

    src_s3 = rasterio.open('NETCDF:"{}":SDR_Oa17'.format(s3_file))
    width, height = src_s3.width, src_s3.height


    with rasterio.open(os.path.join(modis_raster+'merged/', merged_file)) as src:
        transform, width, height = calculate_default_transform(
            src_crs=src.crs, dst_crs=dst_crs, width=src.width, height=src.height, left=src.bounds.left,
            bottom=src.bounds.bottom, right=src.bounds.right, top=src.bounds.top, resolution=src_s3.res)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height#,
            #'nodata': NODATA
        })

        with rasterio.open(os.path.join(modis_raster+'reprojected/', merged_file),
                           'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    #src_nodata=NODATA,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    #dst_nodata=NODATA,
                    resampling=resampling)

    with rasterio.open(os.path.join(modis_raster+'reprojected/', merged_file)) as src:
        window = from_bounds(top=src_s3.bounds.top, bottom=src_s3.bounds.bottom,
                             left=src_s3.bounds.left, right=src_s3.bounds.right,
                             transform=src.transform)
        out_image = src.read(1, window=window)
        #out_image[out_image>1.6] = NODATA
        #out_image[out_image<-0.01] = NODATA
        out_meta = src.meta.copy()
        out_meta.update({"width":src_s3.width, "height":src_s3.height, "transform":src_s3.transform})

        output_file = '{}_{}_{}_{}_{}_wgs84.tif'.format(product, layer, tile_name, date, resampling_method)

        with rasterio.open(os.path.join(output_path.format(tile = tile_name , layer = layer ), output_file), "w", **out_meta) as dest:
            dest.write(np.expand_dims(out_image, axis=0))