# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import numpy as np
import os
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import plotting_extent
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd
import seaborn as sns
# import gdal
#plt.ion()

# def write_file(arr, src_file, outfile='test', no_data_value=-32768, is_complex=True, type='rasterio'):
#     """
#     Write image files in TIF format
#     :param arr: Image array to write
#     :param src_file: Original image file for retrieving affine transformation parameters
#     :param outfile: Output file path
#     :param no_data_value: No data value to be set
#     :param is_complex: If true, write complex image array in two separate bands
#     :return: None
#     """
#
#     driver = gdal.GetDriverByName("GTiff")
#     if is_complex:
#         out = driver.Create(outfile + ".tif", arr.shape[1], arr.shape[0], 2, gdal.GDT_Float32)
#     else:
#         out = driver.Create(outfile + ".tif", arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
#     proj = src_file.crs
#     transform = src_file.transform
#     if type == 'gdal':
#         proj = src_file.GetProjection()
#         transform = src_file.GetGeoTransform()
#     out.SetProjection(proj)
#     out.SetGeoTransform(transform)
#     out.GetRasterBand(1).SetNoDataValue(no_data_value)
#     if is_complex:
#         arr[np.isnan(arr)] = no_data_value + no_data_value * 1j
#         out.GetRasterBand(2).SetNoDataValue(no_data_value)
#         out.GetRasterBand(1).WriteArray(arr.real)
#         out.GetRasterBand(2).WriteArray(arr.imag)
#     else:
#         arr[np.isnan(arr)] = no_data_value
#         out.GetRasterBand(1).WriteArray(arr)
#     out.FlushCache()


sns.set(font_scale=1.5)
os.chdir('/Users/smxnv/Documents/Data')
az_watershed = gpd.read_file('watersheds/az_merged/az_watershed.shp')
az_watershed = mapping(az_watershed['geometry'][0])
az_data = rio.open('cropscape/CDL_2015_clip_20190812153756_568423369/CDL_2015_clip_20190812153756_568423369.tif')
az_crop, az_affine = mask(az_data, [az_watershed], crop=True)
az_extent = plotting_extent(az_crop[0], az_affine)

# fig, ax = plt.subplots(figsize=(10, 8))
# az_plot = ax.imshow(az_crop[0], extent=az_extent)
# ax.set_title("Cropped Raster Dataset")
# ax.set_axis_off()
# fig.colorbar(az_plot)
# plt.show()
az_crop = np.squeeze(az_crop)
print(az_crop.shape)
print(np.max(az_crop))
# with rio.open(
#     '../Output/AZ_Crop.tif',
#     'w',
#     driver='GTiff',
#     height=az_crop.shape[0],
#     width=az_crop.shape[1],
#     dtype=az_crop.dtype,
#     crs=az_data.crs,
#     count=1,
#     transform=az_data.transform,
# ) as dst:
#     dst.write(az_crop, 1)
