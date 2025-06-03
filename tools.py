import ee
import geemap
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import models.unet as unet


def maskS2clouds(image):
    qa = image.select('QA60')
    
    # Bits 10 and 11 are clouds and cirrus, respectively
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    
    # Both flags should be set to zero, indicating clear conditions
    mask = (qa.bitwiseAnd(cloudBitMask).eq(0)
            .And(qa.bitwiseAnd(cirrusBitMask).eq(0)))
    
    return image.updateMask(mask)

def maskLandSurface(image):
    green = image.select('B3') 
    nir = image.select('B8')
    swir1 = image.select('B11')
    swir2 = image.select('B12')
    
    awei = green.multiply(4).subtract(swir1.multiply(4)).subtract(
           nir.multiply(0.25).add(swir2.multiply(2.75)))
    waterMask = awei.gt(0.0)
    return image.updateMask(waterMask)

def LandCoverFilter2(image):
    #mask = (ee.ImageCollection('COPERNICUS/Landcover/100m/Proba-V/Global')
    landcover = (ee.ImageCollection('COPERNICUS/Landcover/100m/Proba-V/Global')
           .mosaic()
           .select('discrete_classification'))
    
    # # Create a complex mask:
    # # Keep pixels where:
    # # 1. Land cover value equals 200 OR
    # # 2. Land cover value is NOT between 20 and 126 (inclusive)
    # land_mask = mask.eq(200).And((mask.gte(20).And(mask.lte(126))).Not())
    
    # # Apply the mask to the image and set masked pixels to 0
    # return image.updateMask(land_mask).unmask(0)

    return image.addBands(landcover.rename('landcover_classification'))

def Image2Aarray(collection, region, scale=10):
    """Convert an image collection to a 4D array (image, band, y, x)"""
    image_count = collection.size().getInfo()
    band_names = collection.first().bandNames().getInfo()
    
    # Create a list to store arrays
    image_arrays = []
    
    # Get dates for labeling
    dates = []
    
    # Convert each image to array
    images = collection.toList(image_count)
    for i in range(image_count):
        image = ee.Image(images.get(i))
        array = geemap.ee_to_numpy(image, region=region, scale=scale)
        
        # Get acquisition date
        date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        dates.append(date)
        
        image_arrays.append(array)
    
    return np.stack(image_arrays), dates, band_names

def GridImages2Array(image, grid_gdf):
    grid_arrays = {}
    grid_coords = {}
    for cell_idx, row in grid_gdf.iterrows():
      geometry = row.geometry
      bounds = geometry.bounds
      cell_geometry = ee.Geometry.Rectangle(bounds)
      
      # Get coordinates for later reference
      grid_coords[cell_idx] = {
          'minx': bounds[0],
          'miny': bounds[1],
          'maxx': bounds[2],
          'maxy': bounds[3],
          'geometry': geometry
      }
      
      # Clip the image to the grid cell
      cell_image = image.clip(cell_geometry).toFloat()
      
      # Export as array (you can choose specific bands if needed)
      # Here we're getting all 11 bands
      cell_array = geemap.ee_to_numpy(
          cell_image, 
          region=cell_geometry,
          bands=image.bandNames().getInfo()
      )
      #cell_array = np.where(np.isinf(cell_array), 0., cell_array)
      grid_arrays[cell_idx] = np.transpose(cell_array, (2,0,1))
    
    return grid_arrays, grid_coords
