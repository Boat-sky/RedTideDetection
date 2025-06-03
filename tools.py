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
    
def z_score_norm(img):
        normalized = np.zeros_like(img, dtype=np.float32)
        mean_vals = [442.82249157627206, 420.2092339188718, 237.88048782700525, 241.14693390215567, 188.6111751669924, 185.5185900821598, 156.0033714034493, 158.70546912717577]
        sd_vals = [424.30795110238336, 403.5894175859025, 309.6133615288345, 313.5777765725138, 269.1745800185496, 257.17273952234876, 254.55214828171793, 234.00751552086538]
        
        for i in range(img.shape[0]):
            normalized[i] = (img[i] - mean_vals[i]) / sd_vals[i]

        return normalized

def img_preprocessing(img):
    # if img.dtype == "uint8":
    #     img = img / 255.
    # elif img.dtype == "uint16":
    #     img = img / 65535.

    img = z_score_norm(img)
    return img

def DetectOneGrid(input_img, model):
  img = input_img[:-1, :, :].copy()
  colap_size = 20
  crop_size = 512
  img = torch.from_numpy(img_preprocessing(img)).float()
  c, h, w = img.shape
  sum_prop = torch.zeros(h, w, device=device)
  predict_count = torch.zeros(h, w, device=device)

  for i in range(0, h, crop_size-colap_size):
    if i + crop_size > h:
        i = h - crop_size
    for j in range(0, w, crop_size-colap_size):
        if j + crop_size > w:
            j = w - crop_size

        cropped_img = img[:, i:i+crop_size, j:j+crop_size]

        tensor_img = torch.tensor(cropped_img).float().to(device)
        tensor_img = tensor_img.unsqueeze(0)

        with torch.no_grad():
            probs = model(tensor_img)

        sum_prop[i:i+crop_size, j:j+crop_size] += probs.squeeze(0).squeeze(0)
        predict_count[i:i+crop_size, j:j+crop_size] += 1

  big_img_probs = sum_prop / predict_count
  big_img_mask = (big_img_probs > 0.5).float().cpu().numpy()
  return big_img_mask

def DetectRedTide(grid_arrays, model):
  model.to(device)
  model.eval()

  result_arrays = {}
  for k, v in grid_arrays.items():
    # if 200 in v[10]:
    #   black_array = np.zeros(v[0].shape)
    #   result_arrays[k] = black_array
    # else:
    #   result_arrays[k] = DetectOneGrid(v, model)
    prediction = DetectOneGrid(np.where(np.isinf(v), 0., v), model)
    inf_mask = np.isinf(v).any(axis=0)

    prediction[inf_mask] = 0
    result_arrays[k] = prediction
  return result_arrays
def merge_grid_arrays_variable_size(result_arrays, grid_coords, target_resolution=None):
    all_minx = min(coord['minx'] for coord in grid_coords.values())
    all_miny = min(coord['miny'] for coord in grid_coords.values())
    all_maxx = max(coord['maxx'] for coord in grid_coords.values())
    all_maxy = max(coord['maxy'] for coord in grid_coords.values())

    # Calculate the resolution for each grid cell
    cell_resolutions = {}
    for cell_idx, array in result_arrays.items():
        coords = grid_coords[cell_idx]
        height, width = array.shape

        x_res = (coords['maxx'] - coords['minx']) / width
        y_res = (coords['maxy'] - coords['miny']) / height

        cell_resolutions[cell_idx] = (x_res, y_res)

    # Calculate average resolution if target resolution is not provided
    if target_resolution is None:
        avg_x_res = np.mean([res[0] for res in cell_resolutions.values()])
        avg_y_res = np.mean([res[1] for res in cell_resolutions.values()])
        target_resolution = (avg_x_res, avg_y_res)

    x_res, y_res = target_resolution
    #print(f"Using target resolution: {x_res:.6f}, {y_res:.6f}")

    # Calculate the size of the final array
    total_width = int(round((all_maxx - all_minx) / x_res))
    total_height = int(round((all_maxy - all_miny) / y_res))

    #print(f"Creating merged array of shape: {total_height} x {total_width}")

    # Create an empty array to hold the merged result
    merged_array = np.zeros((total_height, total_width), dtype=np.uint8)

    # Place each grid array into the correct position in the merged array
    for cell_idx, array in result_arrays.items():
        coords = grid_coords[cell_idx]
        orig_height, orig_width = array.shape

        # Calculate the real-world dimensions of this cell
        cell_width_geo = coords['maxx'] - coords['minx']
        cell_height_geo = coords['maxy'] - coords['miny']

        # Calculate the target dimensions in pixels based on target resolution
        target_width = int(round(cell_width_geo / x_res))
        target_height = int(round(cell_height_geo / y_res))

        # Resize the array if needed
        if target_width != orig_width or target_height != orig_height:
            # Calculate zoom factors
            zoom_x = target_width / orig_width
            zoom_y = target_height / orig_height

            # Use order=0 for nearest neighbor interpolation (good for binary data)
            resized_array = zoom(array, (zoom_y, zoom_x), order=0)

            # Make sure the resized array is binary (0s and 1s only)
            resized_array = (resized_array > 0.5).astype(np.uint8)
        else:
            resized_array = array

        # Calculate where this grid should be placed in the merged array
        start_x = int(round((coords['minx'] - all_minx) / x_res))
        # Flip y-axis (since array origin is top-left but geo coordinates are bottom-left)
        start_y = int(round((all_maxy - coords['maxy']) / y_res))

        # Get the dimensions of the resized array
        res_height, res_width = resized_array.shape

        # Ensure we don't go out of bounds
        end_x = min(start_x + res_width, total_width)
        end_y = min(start_y + res_height, total_height)

        # Trim the resized array if needed
        trim_width = end_x - start_x
        trim_height = end_y - start_y

        print(f"Cell {cell_idx}: Placing array of shape {res_height}x{res_width} " +
              f"at position ({start_y}:{end_y}, {start_x}:{end_x})")

        # Place the resized and trimmed array in the correct position
        try:
            merged_array[start_y:end_y, start_x:end_x] = resized_array[:trim_height, :trim_width]
        except ValueError as e:
            #print(f"Error placing array for cell {cell_idx}: {e}")
            #print(f"Array shape: {resized_array.shape}, Trim dimensions: {trim_height}x{trim_width}")
            #print(f"Target position: {start_y}:{end_y}, {start_x}:{end_x}")
            #print(f"Merged array shape: {merged_array.shape}")

    # Return the merged array and the extent
    return merged_array, (all_minx, all_miny, all_maxx, all_maxy)

def run_program(start_date, end_date):
    lon, lat = 100.50, 13.35
    point = ee.Geometry.Point([lon, lat])
    aoi = point.buffer(3000)

    s2_l2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterDate(start_date, end_date)
    .filterBounds(aoi)
    #.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25))
    .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                        'B8', 'B8A', 'B11', 'B12', 'QA60'])
    )
    
    s2_l2_masked = s2_l2.map(maskS2clouds).map(maskLandSurface).map(LandCoverFilter2)
    
    last_image = s2_l2_masked.sort('system:time_start', False).first()

    half_height = 10000
    half_width =  45000
    rectangle = ee.Geometry.Rectangle([
        lon - half_width/111000,  # แปลงจากระยะทางเป็นองศา
        lat - half_height/111000,
        lon + half_width/111000,
        lat + half_height/111000
    ])
    
    grid = geemap.fishnet(rectangle, h_interval=0.08, v_interval=0.08)
    grid_gdf = geemap.ee_to_gdf(grid)

    bandsKeep = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
                     'B8', 'B8A', 'B11', 'B12', 'landcover_classification']
    
    s2_clipped = last_image.clip(rectangle)
    final_image = s2_clipped.select(bandsKeep)
    print('***** Loading Satellite Image *****')
    grid_arrays, grid_coords = GridImages2Array(final_image, grid_gdf)
    
    print('***** Loading Trained Model *****')
    model_path = hf_hub_download(
    repo_id="Boba-45/red-tide-model",
    #filename="ver2_unet__epoch_5.pth"
    filename="ver2_transunet_epoch_61.pth"
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    n_classes=1
    logits = True
    #model = unet.UNet(in_dim=8, n_classes=n_classes, logits=logits)
    vit_name = 'R50-ViT-B_16'
    n_skip = 3
    img_size = 512
    vit_patches_size = 16
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = 1
    config_vit.n_skip = n_skip
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    model = transunet.VisionTransformer(config=config_vit, img_size=img_size, num_classes=1)
    
    model.load_state_dict(checkpoint['model_state_dict'])

    print('***** Predicting Result *****')
    result_arrays = DetectRedTide(grid_arrays, model)
    merged_array, extent = merge_grid_arrays_variable_size(result_arrays, grid_coords)
