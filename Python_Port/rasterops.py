# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import plotting_extent
from rasterio.mask import mask
from rasterio.enums import Resampling as res
from rasterio.warp import reproject, Resampling
from shapely.geometry import mapping
import geopandas as gpd
import numpy as np
import gdal
from glob import glob
# import seaborn as sns
# plt.ion()


def write_raster(raster_data, raster_file, transform, outfile_path, no_data_value=0):
    """
    Write raster file in GeoTIFF format
    :param raster_data: Raster data to be written
    :param raster_file: Original raster file containing geo-coordinates
    :param transform: Affine transformation matrix
    :param outfile_path: Outfile file path
    :param no_data_value: No data value for raster image
    :return: None
    """

    with rio.open(
            outfile_path,
            'w',
            driver='GTiff',
            height=raster_data.shape[0],
            width=raster_data.shape[1],
            dtype=raster_data.dtype,
            crs=raster_file.crs,
            transform=transform,
            count=raster_file.count,
            nodata=no_data_value
    ) as dst:
        dst.write(raster_data, raster_file.count)


def crop_raster(input_raster_file, input_mask_path, outfile_path, plot_fig=False, plot_title=""):
    """
    Crop raster data based on given shapefile
    :param input_raster_file: Input raster dataset path
    :param input_mask_path: Shapefile path
    :param outfile_path: Output file path (only tiff file)
    :param plot_fig: If true, then cropped raster data is plotted
    :param plot_title: Plot title to display
    :return: Cropped raster dataset
    """

    shape_file = gpd.read_file(input_mask_path)
    shape_file_geom = mapping(shape_file['geometry'][0])
    raster_file = rio.open(input_raster_file)
    raster_crop, raster_affine = mask(raster_file, [shape_file_geom])
    shape_extent = plotting_extent(raster_crop[0], raster_affine)
    raster_crop = np.squeeze(raster_crop)
    write_raster(raster_crop, raster_file, transform=raster_file.transform, outfile_path=outfile_path)
    if plot_fig:
        fig, ax = plt.subplots(figsize=(10, 8))
        raster_plot = ax.imshow(raster_crop[0], extent=shape_extent)
        ax.set_title(plot_title)
        ax.set_axis_off()
        fig.colorbar(raster_plot)
        plt.show()
    return raster_crop


def reclassify_raster(input_raster_file, class_dict, outfile_path):
    """
    Reclassify raster data based on given class dictionary
    :param input_raster_file: Input raster file path
    :param class_dict: Classification dictionary containing (from, to) as keys and "becomes" as value
    :param outfile_path: Output file path (only tiff file)
    :return: Reclassified raster
    """

    raster_file = rio.open(input_raster_file)
    raster_data = raster_file.read(1)
    for key in class_dict.keys():
        raster_data[np.logical_and(raster_data >= key[0], raster_data <= key[1])] = class_dict[key]
    write_raster(raster_data, raster_file, transform=raster_file.transform, outfile_path=outfile_path)
    raster_file.close()
    return raster_data


def resample_raster(input_raster_file, outfile_path, resampling_factor=3, resampling_func=res.mode, downsample=True):
    """
    Resample raster data
    :param input_raster_file: Input raster file path
    :param outfile_path: Output file path
    :param resampling_factor: Resampling factor
    :param resampling_func: Resampling function
    :param downsample: Default true for downsampling, else upsampling
    :return: Resampled raster
    """

    raster_file = rio.open(input_raster_file)
    if downsample:
        height, width = raster_file.height // resampling_factor, raster_file.width // resampling_factor
    else:
        height, width = raster_file.height * resampling_factor, raster_file.width * resampling_factor
    new_shape = (1, height, width, raster_file.count)
    raster_data = raster_file.read(out_shape=new_shape, resampling=resampling_func)
    raster_data = np.squeeze(raster_data)
    transform_factor = np.array([resampling_factor, 1, 1, 1, resampling_factor, 1, 1, 1, 1])
    transform_matrix = np.array(raster_file.transform)
    if downsample:
        transform_matrix = transform_matrix * transform_factor
    else:
        transform_matrix = transform_matrix / transform_factor
    transform_matrix = transform_matrix.tolist()[:6]
    write_raster(raster_data, raster_file, transform=transform_matrix, outfile_path=outfile_path)
    return raster_data


def stack_rasters(input_dir, pattern):
    """
    Create a stack containing several rasters
    :param input_dir: Input directory containing rasters
    :param pattern: File pattern to search for, the file names should end with _mmmyy.tif
    :return: Stack of rasters stored as a dictionary containing (month, year) as keys and rasterio files as values
    """

    raster_stack = {}
    months = {'jan': 'January', 'feb': 'February', 'mar': 'March', 'apr': 'April', 'may': 'May', 'jun': 'June',
              'jul': 'July', 'aug': 'August', 'sep': 'September', 'oct': 'October', 'nov': 'November',
              'dec': 'December'}
    for raster_file in glob(input_dir + '/' + pattern):
        month_yr = raster_file[raster_file.rfind('_') + 1: raster_file.rfind('.')]
        month, year = months[month_yr[:3].lower()], int('20' + month_yr[3:])
        raster_stack[(month, year)] = rio.open(raster_file)
    return raster_stack


def apply_raster_stack_arithmetic(raster_stack, outfile_path, ops='sum'):
    """
    Apply arithmetric operations on a raster stack
    :param raster_stack: Input raster stack stored as a dictionary
    :param outfile_path: Output file path
    :param ops: Specified operation, default is sum, others include sub and mul
    :return: Resultant raster
    """

    raster_file_list = list(raster_stack.values())
    result_arr = raster_file_list[0].read(1)
    for raster_file in raster_file_list[1:]:
        raster_arr = raster_file.read(1)
        if ops == 'sum':
            result_arr = result_arr + raster_arr
        elif ops == 'sub':
            result_arr = result_arr - raster_arr
        else:
            result_arr = result_arr * raster_arr
    write_raster(result_arr, raster_file_list[0], transform=raster_file_list[0].transform, outfile_path=outfile_path)
    return result_arr


def reproject_raster(src_raster_file, dst_raster_file, outfile_path, resampling=gdal.GRA_NearestNeighbour):
    """
    Reproject one raster to another
    :param src_raster_file: Source raster file
    :param dst_raster_file: Destination raster file
    :param outfile_path: Output file path
    :param resampling: Resampling technique, nearest neighbor is the default
    :return: Reprojected raster
    """

    src_raster_file = gdal.Open(src_raster_file)
    dst_raster_file = gdal.Open(dst_raster_file)
    transform = dst_raster_file.GetGeoTransform()
    xRes, yRes = transform[1], transform[5]
    gdal.Warp(outfile_path, src_raster_file, xRes=xRes, yRes=yRes, resampleAlg=resampling,
              dstSRS=dst_raster_file.GetProjection())
