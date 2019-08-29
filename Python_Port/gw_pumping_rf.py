# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

from rasterio.enums import Resampling as res
import gdal

from Python_Port import rasterops as rops
from Python_Port import vectorops as vops

NO_DATA_VALUE = 0

input_dir = '/Users/smxnv/Documents/Data/'
output_dir = '/Users/smxnv/Documents/Output/'
# az_raster_path = input_dir + 'cropscape/CDL_2015_clip_20190812153756_568423369/' \
#                              'CDL_2015_clip_20190812153756_568423369.tif'
# az_ws_path = input_dir + 'watersheds/az_merged/az_watershed.shp'
# az_crop_file = output_dir + 'az_crop.tif'
# az_cropped = rops.crop_raster(az_raster_path, az_ws_path, az_crop_file)
#
# az_class_dict = {(0, 59.5): 1,
#                  (67.5, 75.5): 1,
#                  (203.5, 250): 1,
#                  (110.5, 111.5): 2,
#                  (120.5, 124.5): 3,
#                  (60.5, 61.5): NO_DATA_VALUE,
#                  (130.5, 195.5): NO_DATA_VALUE
#                  }

# az_reclass_file = output_dir + 'az_reclass.tif'
# az_reclass = rops.reclassify_raster(az_crop_file, az_class_dict, az_reclass_file)
#
# ks_raster_path = input_dir + 'cropscape/polygonclip_20190306140312_392696635/' \
#                              'CDL_2015_clip_20190306140312_392696635.tif'
# ks_ws_path = input_dir + 'watersheds/ks_merged/ks_watershed.shp'
# ks_crop_file = output_dir + 'ks_crop.tif'
ks_crop_file = input_dir + 'cropscape/polygonclip_20190306140312_392696635/' \
                              'CDL_2015_clip_20190306140312_392696635.tif'
# ks_cropped = rops.crop_raster(ks_raster_path, ks_ws_path, ks_crop_file)

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
ks_reclass_file_2 = output_dir + 'ks_reclass2.tif'
# ks_reclass2 = rops.resample_raster(ks_reclass_file, ks_reclass_file_2)
ks_reclass2 = rops.gdal_warp_syscall(ks_reclass_file, ks_reclass_file_2, resampling_func=gdal.GRA_NearestNeighbour)

# ET_stack = rops.stack_rasters(input_dir + 'ET_precip', pattern='ET*')
# P_stack = rops.stack_rasters(input_dir + 'ET_precip', pattern='precip*')
#
# demand_file = output_dir + 'demand.tif'
# demand_all = rops.apply_raster_stack_arithmetic(ET_stack, outfile_path=demand_file)
# ks_resamp_file = output_dir + 'ks_resamp.tif'
# ks_resamp = rops.reproject_raster(ks_reclass_file, demand_file, outfile_path=ks_resamp_file)
#
# demand_flt_file = output_dir + 'demand_flt.tif'
# demand_all = rops.apply_raster_filter(ks_resamp_file, demand_file, outfile_path=demand_flt_file, flt_values=(1, ))
#
# gw_file = input_dir + '2015_smoothed/wuse_st_data/wuse_density_0515_5mile_clip.img'
# gw_pumping_file, gw_arr = rops.get_gw_pumping(gw_file)
# ks_ws_reproj_path = output_dir + 'ks_ws_reproj.shp'
# ks_watershed2 = vops.reproject_vector(ks_ws_path, outfile_path=ks_ws_reproj_path, ref_file=gw_file)
# demand_all_reproj_file = output_dir + 'demand_all_reproj.tif'
# demand_all_reproj = rops.reproject_raster(demand_flt_file, gw_file, outfile_path=demand_all_reproj_file)
#
# da_res_file = output_dir + 'da_res.tif'
# gw_res_file = output_dir + 'gw_res.tif'
# ag=5
#
# da_res = rops.resample_raster(demand_all_reproj_file, outfile_path=da_res_file, resampling_factor=ag)
# gw_res = rops.resample_raster(gw_file, outfile_path=gw_res_file, resampling_factor=ag)
#
# water_file = output_dir + 'water.tif'
# water = rops.apply_raster_filter2(ks_resamp_file, outfile_path=water_file)
# water2_file = output_dir + 'water2.tif'
# water2 = rops.reproject_raster(water_file, gw_file, outfile_path=water2_file)
# water3_file = output_dir + 'water3.tif'
# water3 = rops.gdal_warp_syscall(water2_file, water3_file, resampling_factor=ag)
# water_flt_file = output_dir + 'water_flt.tif'
# water_flt = rops.apply_gaussian_filter(water3_file, outfile_path=water_flt_file, sigma=5)
#
# urban_file = output_dir + 'urban.tif'
# urban = rops.apply_raster_filter2(ks_resamp_file, outfile_path=urban_file, val=3)
# urban_reproj_file = output_dir + 'urban_reproj.tif'
# urban = rops.reproject_raster(urban_file, gw_file, outfile_path=urban_reproj_file)
# urban_resamp_file = output_dir + 'urban_resamp.tif'
# urban = rops.resample_raster(urban_reproj_file, outfile_path=urban_resamp_file, resampling_factor=ag,
#                              resampling_func=res.average)
# p_all_file = output_dir + 'p_all.tif'
# p_all = rops.apply_raster_stack_arithmetic(P_stack, p_all_file)
# p_all_reproj_file = output_dir + 'p_all_reproj.tif'
# p_all_reproj =  rops.reproject_raster(p_all_file, gw_file, outfile_path=p_all_reproj_file)
# p_all_resample_file = output_dir + 'p_all_reproj_res.tif'
# p_all_reproj_res = rops.resample_raster(p_all_reproj_file, outfile_path=p_all_resample_file, resampling_factor=ag,
#                                         resampling_func=res.average)
