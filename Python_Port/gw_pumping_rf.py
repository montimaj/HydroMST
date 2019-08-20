# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

from Python_Port import rasterops as rops
import numpy as np
NO_DATA_VALUE = 255

input_dir = '/Users/smxnv/Documents/Data/'
output_dir = '/Users/smxnv/Documents/Output/'
az_raster_path = input_dir + 'cropscape/CDL_2015_clip_20190812153756_568423369/' \
                             'CDL_2015_clip_20190812153756_568423369.tif'
az_ws_path = input_dir + 'watersheds/az_merged/az_watershed.shp'
az_crop_file = output_dir + 'az_crop.tif'
az_cropped = rops.crop_raster(az_raster_path, az_ws_path, az_crop_file)

az_class_dict = {(0, 59.5): 1,
                 (67.5, 75.5): 1,
                 (203.5, 250): 1,
                 (110.5, 111.5): 2,
                 (120.5, 124.5): 3,
                 (60.5, 61.5): NO_DATA_VALUE,
                 (130.5, 195.5): NO_DATA_VALUE
                 }
az_reclass_file = output_dir + 'az_reclass.tif'
az_reclass = rops.reclassify_raster(az_crop_file, az_class_dict, az_reclass_file)

ks_raster_path = input_dir + 'cropscape/polygonclip_20190306140312_392696635/' \
                             'CDL_2015_clip_20190306140312_392696635.tif'
ks_ws_path = input_dir + 'watersheds/ks_merged/ks_watershed.shp'
ks_crop_file = output_dir + 'ks_crop.tif'
ks_cropped = rops.crop_raster(ks_raster_path, ks_ws_path, ks_crop_file)

ks_class_dict = {(0, 59.5): 1,
                 (66.5, 77.5): 1,
                 (203.5, 255): 1,
                 (110.5, 111.5): 2,
                 (111.5, 112.5): NO_DATA_VALUE,
                 (120.5, 124.5): 3,
                 (59.5, 61.5): NO_DATA_VALUE,
                 (130.5, 195.5): NO_DATA_VALUE
                 }

ks_reclass_file = output_dir + 'ks_reclass.tif'
ks_reclass = rops.reclassify_raster(ks_crop_file, ks_class_dict, ks_reclass_file)