import ee
import geemap
import numpy as np

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

def ImageCollection2List(s2_collection, aoi, scale=10):
    
    # Get the list of images in the collection
    image_list = s2_collection.toList(s2_collection.size())
    
    # Get the size of the collection
    collection_size = image_list.size().getInfo()
    print(f"Processing {collection_size} images...")
    
    results = []
    
    # Process each image in the collection
    for i in range(collection_size):
        # Get the image from the list
        image = ee.Image(image_list.get(i))
        
        # Get the image date for reference
        image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        print(f"Processing image from {image_date}")
        
        # Select the bands you want to use
        bands_to_select = ['B2', 'B3', 'B4', 'B8']  # Example: RGB + NIR
        image_with_bands = image.select(bands_to_select)
        
        # Convert to NumPy array
        image_array = geemap.ee_to_numpy(
            image_with_bands,
            region=aoi,
            scale=scale
        )
        
        # Skip if the image has no valid data (all zeros or NaN)
        if np.all(image_array == 0) or np.isnan(image_array).any():
            print(f"Skipping image from {image_date} - no valid data")
            continue
        
        # Preprocess the array for your model
        # Reshape: [height, width, channels] -> [channels, height, width]
        input_array = np.transpose(image_array, (2, 0, 1))
        results.append(input_array)
    return results
