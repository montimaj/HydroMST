# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import gdal

from Python_Files import rasterops as rops
from Python_Files import vectorops as vops
from Python_Files import random_forest_regressor as rfr
import numpy as np
from glob import glob
import os


def makedirs(directory_list):
    """
    Create directory for storing files
    :param directory_list: List of directories to create
    :return: None
    """

    for directory_name in directory_list:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)


input_dir = '../Data/'
file_dir = '../Files/'
output_dir = '../Output/'
input_ts_dir = input_dir + 'Time_Series/'
output_shp_dir = file_dir + 'GW_Shapefiles/'
output_gw_raster_dir = file_dir + 'GW_Rasters/'
makedirs([output_dir, output_shp_dir, output_gw_raster_dir])

print('Converting CSV to SHP...')
vops.csvs2shps(input_ts_dir, output_shp_dir, target_crs='epsg:26914', delim='\t', pattern='*.txt')
print('Converting SHP to TIF...')
vops.shps2rasters(output_shp_dir, output_gw_raster_dir)

print('Reprojecting KS Watershed Vector...')
ks_watershed_file = input_dir + 'Watersheds/ks_merged/ks_watershed.shp'
ks_watershed_reproj_dir = file_dir + 'Watersheds/ks_reproj/'
makedirs([ks_watershed_reproj_dir])
ks_watershed_reproj_file = ks_watershed_reproj_dir + 'ks_watershed_reproj.shp'
ref_raster = glob(output_gw_raster_dir + '*.tif')[0]
vops.reproject_vector(ks_watershed_file, outfile_path=ks_watershed_reproj_file, ref_file=ref_raster)
print('Masking Rasters using KS Watershed...')
gw_mask_dir = output_gw_raster_dir + 'Masked/'
makedirs([gw_mask_dir])
rops.crop_rasters(output_gw_raster_dir, input_mask_file=ks_watershed_reproj_file, outdir=gw_mask_dir)

print('Smoothing GW Rasters...')
gw_smoothed_dir = output_gw_raster_dir + 'Smoothed/'
makedirs([gw_smoothed_dir])
rops.smooth_rasters(gw_mask_dir, outdir=gw_smoothed_dir)

print('Reclassifying KS CDL 2015 data...')
na_int = np.iinfo(np.int32).min
na_float = np.finfo(np.float32).min
ks_class_dict = {(0, 59.5): 1,
                 (66.5, 77.5): 1,
                 (203.5, 255): 1,
                 (110.5, 111.5): 2,
                 (111.5, 112.5): na_int,
                 (120.5, 124.5): 3,
                 (59.5, 61.5): na_int,
                 (130.5, 195.5): na_int
                 }

ks_cdl_file = input_dir + 'CDL/CDL_KS_2015.tif'
ks_reclass_dir = file_dir + 'KS_Reclass/'
makedirs([ks_reclass_dir])
ks_reclass_file = ks_reclass_dir + 'ks_reclass.tif'
ks_reclass = rops.reclassify_raster(ks_cdl_file, ks_class_dict, ks_reclass_file)
ks_reclass_file_2 = ks_reclass_dir + 'ks_reclass2.tif'
ks_reclass2 = rops.gdal_warp_syscall(ks_reclass_file, ks_reclass_file_2)

print('Reprojecting rasters...')
raster_reproj_dir = file_dir + 'Reproj_Rasters/'
makedirs([raster_reproj_dir])
ref_raster = glob(gw_mask_dir + '*.tif')[0]
rops.reproject_rasters(input_ts_dir, ref_raster=ref_raster, outdir=raster_reproj_dir)

print('Masking rasters...')
raster_mask_dir = file_dir + 'Masked_Rasters/'
makedirs([raster_mask_dir])
rops.mask_rasters(raster_reproj_dir, ref_raster=ref_raster, outdir=raster_mask_dir)

print('Filtering ET files...')
et_flt_dir = file_dir + 'ET_FLT/'
makedirs([et_flt_dir])
ks_resamp_file = ks_reclass_dir + 'ks_resamp.tif'
ks_reproj_file = ks_reclass_dir + 'ks_reproj.tif'
rops.gdal_warp_syscall(ks_resamp_file, from_raster=ref_raster, outfile_path=ks_reproj_file)
rops.apply_et_filter(raster_mask_dir, outdir=et_flt_dir, ref_raster=ks_reproj_file)

print('Surface Water file...')
water_dir = file_dir + 'SW/'
makedirs([water_dir])
water_file = water_dir + 'water.tif'
water = rops.apply_raster_filter2(ks_reproj_file, outfile_path=water_file)
water_masked = water_dir + 'water_masked.tif'
rops.filter_nans(water_file, ref_raster, outfile_path=water_masked)
water_flt =water_dir + 'water_flt.tif'
rops.apply_gaussian_filter(water_masked, outfile_path=water_flt, sigma=5, normalize=True, ignore_nan=False)

print('Urban file...')
urban_dir = file_dir + 'Urban/'
makedirs([urban_dir])
urban_file = urban_dir + 'urban.tif'
urban = rops.apply_raster_filter2(ks_reproj_file, outfile_path=urban_file, val=3)
urban_masked = urban_dir + 'urban_masked.tif'
rops.filter_nans(urban_file, ref_raster, outfile_path=urban_masked)

print('Random Forest...')
df_file = output_dir + '/df.csv'
df = rfr.create_dataframe(file_dir + 'RF_Data', out_df=df_file)
rf = rfr.rf_regressor(df, output_dir, n_estimators=200, random_state=0, test_size=0.2)

