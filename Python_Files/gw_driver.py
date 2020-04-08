# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import numpy as np
import os
import pandas as pd
from Python_Files import rasterops as rops
from Python_Files import vectorops as vops
from Python_Files import random_forest_regressor as rfr
from glob import glob


def makedirs(directory_list):
    """
    Create directory for storing files
    :param directory_list: List of directories to create
    :return: None
    """

    for directory_name in directory_list:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)


def optimize_hyperparameters(df, test_cases, est_range, f_range, drop_attrs, pred_attr):
    """
    Function for optimizing the primary hyperparameters
    :param df: Data frame
    :param test_cases: Test cases in a list
    :param est_range: Number of estimators range
    :param f_range: Number of max features range
    :param drop_attrs: Attributes to drop during tuning
    :param pred_attr: Target attribute to predict
    :return: None
    """

    ts = []
    for n in range(1, len(test_cases) + 1):
        ts.append('T' + str(n))
    for ne in est_range:
        for nf in f_range:
            for y, t in zip(test_cases, ts):
                if not isinstance(y, range):
                    ty = (y,)
                else:
                    ty = y
                rfr.rf_regressor(df, output_dir, n_estimators=ne, random_state=0, pred_attr=pred_attr,
                                 drop_attrs=drop_attrs, test_year=ty, shuffle=False, plot_graphs=False,
                                 split_yearly=True, bootstrap=True, max_features=nf, test_case=t)


input_dir = '../Data/'
input_dir_2 = '../Data_Summer/'
file_dir = '../Files_New/'
output_dir = '../Output/'
input_ts_dir = input_dir + 'Time_Series_New/'
output_shp_dir = file_dir + 'GW_Shapefiles/'
output_all_shp_dir = file_dir + 'GW_All_Shapefiles/'
output_gw_raster_dir = file_dir + 'GW_Rasters_All/'
makedirs([output_dir, output_shp_dir, output_gw_raster_dir, output_all_shp_dir])


# print('Extracting GW data from GDB...')
# input_gdb_dir = input_dir + 'ks_pd_data_updated2018.gdb'
# attr_name = 'AF_USED'
# year_list = range(2018, 2019)
# vops.extract_gdb_data(input_gdb_dir, attr_name=attr_name, year_list=year_list, outdir=output_all_shp_dir)
#
# print('Converting CSV to SHP...')
# vops.csvs2shps(input_ts_dir, output_shp_dir, target_crs='epsg:26914', delim='\t', pattern='*.txt')
# print('Reprojecting KS Watershed Vector...')
# ks_watershed_file = input_dir + 'Watersheds/ks_merged/ks_watershed.shp'
ks_watershed_reproj_dir = file_dir + 'Watersheds/ks_reproj/'
# makedirs([ks_watershed_reproj_dir])
ks_watershed_reproj_file = ks_watershed_reproj_dir + 'ks_watershed_reproj.shp'
# ref_shp = glob(output_shp_dir + '*.shp')[0]
# vops.reproject_vector(ks_watershed_file, outfile_path=ks_watershed_reproj_file, ref_file=ref_shp, raster=False)
# print('Clipping GW shapefiles...')
clipped_gw_shp_dir = output_all_shp_dir + 'Clipped/'
# makedirs([clipped_gw_shp_dir])
# vops.clip_vectors(output_all_shp_dir, clip_file=ks_watershed_reproj_file, outdir=clipped_gw_shp_dir,
#                   gdal_path='C:/OSGeo4W64/')
# print('Converting SHP to TIF...')
# vops.shps2rasters(clipped_gw_shp_dir, output_gw_raster_dir, xres=5000, yres=5000, smoothing=0,
#                   gdal_path='C:/OSGeo4W64/')
#
# print('Masking Rasters using KS Watershed...')
gw_mask_dir = output_gw_raster_dir + 'Masked/'
# makedirs([gw_mask_dir])
# rops.crop_rasters(output_gw_raster_dir, input_mask_file=ks_watershed_reproj_file, outdir=gw_mask_dir,
#                   gdal_path='C:/OSGeo4W64/')

# print('Updated GW files...This will take significant time as pixelwise operations are performed!!')
updated_gw_dir = gw_mask_dir + 'Updated/'
# makedirs([updated_gw_dir])
# rops.compute_rasters_from_shp(input_raster_dir=gw_mask_dir, input_shp_dir=clipped_gw_shp_dir, outdir=updated_gw_dir,
#                               gdal_path='C:/OSGeo4W64/')

# print('Changing GW units from acreft to mm')
# new_gw_dir = gw_mask_dir + 'Converted/'
# makedirs([new_gw_dir])
# rops.convert_gw_data(updated_gw_dir, new_gw_dir)

# print('Reclassifying KS CDL 2015 data...')
# ks_class_dict = {(0, 59.5): 1,
#                  (66.5, 77.5): 1,
#                  (203.5, 255): 1,
#                  (110.5, 111.5): 2,
#                  (111.5, 112.5): np.int(NO_DATA_VALUE),
#                  (120.5, 124.5): 3,
#                  (59.5, 61.5): np.int(NO_DATA_VALUE),
#                  (130.5, 195.5): np.int(NO_DATA_VALUE)
#                  }
#
# ks_cdl_file = input_dir + 'CDL/CDL_KS_2015.tif'
# ks_reclass_dir = file_dir + 'KS_Reclass/'
# makedirs([ks_reclass_dir])
# ks_reclass_file = ks_reclass_dir + 'ks_reclass.tif'
# ks_reclass = rops.reclassify_raster(ks_cdl_file, ks_class_dict, ks_reclass_file)
# ks_reclass_file_2 = ks_reclass_dir + 'ks_reclass2.tif'
# ks_reclass2 = rops.gdal_warp_syscall(ks_reclass_file, ks_reclass_file_2)
#
# print('Reprojecting rasters...')
# raster_reproj_dir = file_dir + 'Reproj_Rasters_All/'
# makedirs([raster_reproj_dir])
# ref_raster = glob(gw_mask_dir + '*.tif')[0]
# ref_raster = glob('../Files/RF_Data_All/' + '*.tif')[0]
# rops.reproject_rasters(input_ts_dir, ref_raster=ref_raster, outdir=raster_reproj_dir, pattern='*.tif',
#                        gdal_path='C:/OSGeo4W64/')
#
# print('Masking rasters...')
# raster_mask_dir = file_dir + 'Masked_Rasters_All/'
# makedirs([raster_mask_dir])
# rops.mask_rasters(raster_reproj_dir, ref_raster=ref_raster, outdir=raster_mask_dir, pattern='*.tif')
#
# print('Filtering ET files...')
# et_flt_dir = file_dir + 'ET_FLT_All/'
# makedirs([et_flt_dir])
# ks_reproj_file = ks_reclass_dir + 'ks_reproj.tif'
# rops.gdal_warp_syscall(ks_reclass_file, outfile_path=ks_reproj_file, from_raster=ref_raster)
# rops.apply_et_filter(raster_mask_dir, outdir=et_flt_dir, ref_raster1=ks_reproj_file, ref_raster2=ref_raster)
#
# print('Surface Water file...')
# water_dir = file_dir + 'SW/'
# makedirs([water_dir])
# water_file = water_dir + 'water.tif'
# water = rops.apply_raster_filter2(ks_reproj_file, outfile_path=water_file)
# water_masked = water_dir + 'water_masked.tif'
# rops.filter_nans(water_file, ref_raster, outfile_path=water_masked)
# water_flt = water_dir + 'water_flt.tif'
# et_raster = glob(et_flt_dir + '*.tif')[0]
# rops.apply_gaussian_filter(water_masked, ref_file=et_raster, outfile_path=water_flt, sigma=5, normalize=True,
#                            ignore_nan=False)
#
# print('Urban file...')
# urban_dir = file_dir + 'Urban/'
# makedirs([urban_dir])
# urban_file = urban_dir + 'urban.tif'
# urban = rops.apply_raster_filter2(ks_reproj_file, outfile_path=urban_file, val=3)
# urban_masked = urban_dir + 'urban_masked.tif'
# rops.filter_nans(urban_file, ref_raster, outfile_path=urban_masked)
# rops.fill_nans(urban_file, ref_file=et_raster, outfile_path=urban_masked)

# print('Urban Smoothed file...')
# urban_dir = file_dir + 'Urban_Smoothed/'
# makedirs([urban_dir])
# urban_file = urban_dir + 'urban.tif'
# urban = rops.apply_raster_filter2(ks_reproj_file, outfile_path=urban_file, val=3)
# urban_masked = urban_dir + 'urban_masked.tif'
# rops.filter_nans(urban_file, ref_raster, outfile_path=urban_masked)
# urban_flt = urban_dir + 'urban_flt.tif'
# et_raster = glob(et_flt_dir + '*.tif')[0]
# rops.apply_gaussian_filter(urban_masked, ref_file=et_raster, outfile_path=urban_flt, sigma=3, normalize=True,
#                            ignore_nan=False)
#
# print('Agriculture Smoothed file...')
# agri_dir = file_dir + 'Agri_Smoothed/'
# makedirs([agri_dir])
# agri_file = agri_dir + 'agri.tif'
# agri = rops.apply_raster_filter2(ks_reproj_file, outfile_path=agri_file, val=1)
# agri_masked = agri_dir + 'agri_masked.tif'
# rops.filter_nans(agri_file, ref_raster, outfile_path=agri_masked)
# agri_flt = agri_dir + 'agri_flt.tif'
# et_raster = glob(et_flt_dir + '*.tif')[0]
# rops.apply_gaussian_filter(agri_masked, ref_file=et_raster, outfile_path=agri_flt, sigma=3, normalize=True,
#                            ignore_nan=False)

# print('GRACE average...')
# grace_dir = raster_mask_dir + 'GRACE_AVERAGED/'
# makedirs([grace_dir])
# rops.fill_mean_value(raster_mask_dir, outdir=grace_dir)

# print('GRACE Scale...')
# grace_dir = raster_mask_dir + 'GRACE_Scaled/'
# makedirs([grace_dir])
# rops.scale_raster_data(raster_mask_dir, grace_dir, pattern='GRACE*.tif')

# print('GRACE Trend average...')
# grace_dir = raster_mask_dir + 'GRACE_TREND_AVERAGED/'
# makedirs([grace_dir])
# rops.fill_mean_value(raster_mask_dir, outdir=grace_dir, pattern='GRACE_Trend*.tif')

# Dataframe Creation (All the previous steps are for data pre-processing)
print('DataFrame & Random Forest...')
df_file = output_dir + '/raster_df_all.csv'
rf_data_dir = file_dir + 'RF_Data_All/'
df = rfr.create_dataframe(rf_data_dir, out_df=df_file, make_year_col=True, exclude_years=(2019, ))
# df = pd.read_csv(df_file)
drop_attrs = ('YEAR',)
pred_attr = 'GW_KS'

# Hyperparameter Optimization
# n_features = len(df.columns) - len(drop_attrs) - 1
# test_cases = [2014, 2012, range(2012, 2017)]
# est_range = range(100, 601, 100)
# f_range = range(1, n_features + 1)
# optimize_hyperparameters(test_cases, est_range, f_range)

# Final Model and Prediction
plot_dir = output_dir + 'Partial_Plots/PDP_Data/'
makedirs([plot_dir])
rf_model = rfr.rf_regressor(df, output_dir, n_estimators=500, random_state=0, pred_attr=pred_attr,
                            drop_attrs=drop_attrs, test_year=range(2011, 2020), shuffle=False, plot_graphs=False,
                            plot_3d=False, split_yearly=True, bootstrap=True, plot_dir=plot_dir, max_features=3,
                            load_model=True)
pred_years = range(2002, 2020)
pred_out_dir = output_dir + 'Predicted_Rasters_All/'
makedirs([pred_out_dir])
rfr.predict_rasters(rf_model, pred_years=pred_years, drop_attrs=drop_attrs, out_dir=pred_out_dir,
                    actual_raster_dir=rf_data_dir, pred_attr=pred_attr, only_pred=False, exclude_years=(2019,))
crop_dir = output_dir + 'Cropped_Rasters_All/'
makedirs([crop_dir])
rops.crop_multiple_rasters(rf_data_dir, outdir=crop_dir, input_shp_file=file_dir + 'Final_Mask/crop.shp',
                           pattern='GW*.tif')
rops.crop_multiple_rasters(pred_out_dir, outdir=crop_dir, input_shp_file=file_dir + 'Final_Mask/crop.shp')
