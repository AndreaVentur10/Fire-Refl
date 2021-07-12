from osgeo import gdal
import rasterio


def GenerateAttribute(dem, path, attribute):
    gdal.DEMProcessing(path+attribute+'.tif', dem, attribute)
    with rasterio.open(path+attribute+'.tif') as dataset:
        data = dataset.read(1)
    return data


def GenerateSlope(dem , path):
    gdal.DEMProcessing(path+'slope.tif', dem, 'slope')
    with rasterio.open(path+'slope.tif') as dataset:
        dataset.read(1)


def GenerateAspect(dem, path):
    gdal.DEMProcessing(path+'aspect.tif', dem, 'aspect')
    with rasterio.open(path+'aspect.tif') as dataset:
        dataset.read(1)


def GenerateHillShade(dem, path):
    gdal.DEMProcessing(path+'hillshade.tif', dem, 'hillshade')
    with rasterio.open(path+'hillshade.tif') as dataset:
        dataset.read(1)
