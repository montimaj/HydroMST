# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import plotting_extent
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd
import numpy as np
import gdal
import astropy.convolution as apc
from glob import glob
import scipy.ndimage.filters as flt
import subprocess

NO_DATA_VALUE = np.finfo(np.float32).min


def read_raster_as_arr(raster_file, band=1, get_file=True, rasterio_obj=False, change_dtype=True):
    """
    Get raster array
    :param raster_file: Input raster file path
    :param band: Selected band to read (Default 1)
    :param get_file: Get rasterio object file if set to True
    :param rasterio_obj: Set true if raster_file is a rasterio object
    :param change_dtype: Change raster data type to float if true
    :return: Raster numpy array and rasterio object file (get_file=True and rasterio_obj=False)
    """

    if not rasterio_obj:
        raster_file = rio.open(raster_file)
    else:
        get_file = False
    raster_arr = raster_file.read(band)
    if change_dtype:
        raster_arr = raster_arr.astype(np.float32)
        raster_arr[raster_arr == raster_file.nodata] = np.nan
    if get_file:
        return raster_arr, raster_file
    return raster_arr


def write_raster(raster_data, raster_file, transform, outfile_path, no_data_value=NO_DATA_VALUE,
                 ref_file=None):
    """
    Write raster file in GeoTIFF format
    :param raster_data: Raster data to be written
    :param raster_file: Original rasterio raster file containing geo-coordinates
    :param transform: Affine transformation matrix
    :param outfile_path: Outfile file path
    :param no_data_value: No data value for raster (default float32 type is considered)
    :param ref_file: Write output raster considering parameters from reference raster file
    :return: None
    """
    if ref_file:
        raster_file = rio.open(ref_file)
        transform = raster_file.transform
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
    write_raster(raster_crop, raster_file, transform=raster_file.transform, outfile_path=outfile_path,
                 no_data_value=raster_file.nodata)
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
    Reclassify raster data based on given class dictionary (left exclusive, right inclusive)
    :param input_raster_file: Input raster file path
    :param class_dict: Classification dictionary containing (from, to) as keys and "becomes" as value
    :param outfile_path: Output file path (only tiff file)
    :return: Reclassified raster
    """

    raster_arr, raster_file = read_raster_as_arr(input_raster_file, change_dtype=False)
    for key in class_dict.keys():
        raster_arr[np.logical_and(raster_arr > key[0], raster_arr <= key[1])] = class_dict[key]
    raster_arr = raster_arr.astype(np.float32)
    raster_arr[raster_arr == 0] = NO_DATA_VALUE
    write_raster(raster_arr, raster_file, transform=raster_file.transform, outfile_path=outfile_path)
    return raster_arr


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
    result_arr = read_raster_as_arr(raster_file_list[0], rasterio_obj=True)
    for raster_file in raster_file_list[1:]:
        raster_arr = read_raster_as_arr(raster_file, rasterio_obj=True)
        if ops == 'sum':
            result_arr = result_arr + raster_arr
        elif ops == 'sub':
            result_arr = result_arr - raster_arr
        else:
            result_arr = result_arr * raster_arr
    result_arr[np.isnan(result_arr)] = NO_DATA_VALUE
    write_raster(result_arr, raster_file_list[0], transform=raster_file_list[0].transform, outfile_path=outfile_path)
    return result_arr


def apply_raster_filter(raster_file1, raster_file2, outfile_path, flt_values=(), new_value=0):
    """
    Apply filter on raster 1 and set corresponding values in raster 2. The rasters must be aligned properly beforehand.
    :param raster_file1: Apply filter values to this raster file
    :param raster_file2: Change values of this raster file with respect to raster 1 filter
    :param outfile_path: Output file path
    :param flt_values: Tuple of filter values
    :param new_value: Replacement value in the new raster
    :return: Modified raster array
    """

    rf1_arr, raster_file1 = read_raster_as_arr(raster_file1)
    rf2_arr, raster_file2 = read_raster_as_arr(raster_file2)
    for val in flt_values:
        rf2_arr[np.where(np.logical_and(~np.isnan(rf1_arr), rf1_arr != val))] = new_value
    rf2_arr[np.isnan(rf2_arr)] = NO_DATA_VALUE
    write_raster(rf2_arr, raster_file2, transform=raster_file2.transform, outfile_path=outfile_path)
    raster_file1.close()
    raster_file2.close()


def apply_raster_filter2(input_raster_file, outfile_path, val=2):
    """
    Extract selected value from raster
    :param input_raster_file: Input raster file
    :param outfile_path: Output raster file
    :param val: Value to be selected from raster
    :return: Raster numpy array
    """

    raster_arr, input_raster_file = read_raster_as_arr(input_raster_file)
    raster_arr[raster_arr != val] = input_raster_file.nodata
    raster_arr[raster_arr == input_raster_file.nodata] = NO_DATA_VALUE
    write_raster(raster_arr, input_raster_file, transform=input_raster_file.transform, outfile_path=outfile_path)
    return raster_arr


def get_gw_pumping(gw_raster_file):
    """
    Groundwater pumping raster data in mm
    :param gw_raster_file: Input GW raster file
    :return: GW rasterio file and numpy array
    """

    gw_arr, gw_raster_file = read_raster_as_arr(gw_raster_file)
    gw_arr *= 1233.48 * 1000. / 2.59e+6
    return gw_raster_file, gw_arr


def filter_nans(raster_file, ref_file, outfile_path):
    """
    Set nan considering reference file to a raster file
    :param raster_file: Input raster file
    :param ref_file: Reference file
    :param outfile_path: Output file path
    :return: Modified raster array
    """

    raster_arr, raster_file = read_raster_as_arr(raster_file)
    ref_arr = read_raster_as_arr(ref_file, get_file=False)
    raster_arr[np.isnan(ref_arr)] = NO_DATA_VALUE
    write_raster(raster_arr, raster_file, transform=raster_file.transform, outfile_path=outfile_path)


def apply_gaussian_filter(input_raster_file, outfile_path, sigma=3, normalize=False, ignore_nan=True):
    """
    Apply a gaussian filter over a raster image
    :param input_raster_file: Input raster file
    :param outfile_path: Output file path
    :param sigma: Standard Deviation for gaussian kernel (default 3)
    :param normalize: Set true to normalize the filtered raster at the end
    :param ignore_nan: Set true to ignore nan values during convolution
    :return: Gaussian filtered raster
    """

    raster_arr, input_raster_file = read_raster_as_arr(input_raster_file)
    if ignore_nan:
        gaussian_kernel = apc.Gaussian2DKernel(x_stddev=sigma, x_size=3 * sigma, y_size=3 * sigma)
        raster_arr_flt = apc.convolve(raster_arr, gaussian_kernel, preserve_nan=True)
        raster_arr[np.isnan(raster_arr)] = NO_DATA_VALUE
    else:
        raster_arr[np.isnan(raster_arr)] = 0
        raster_arr_flt = flt.gaussian_filter(raster_arr, sigma=sigma, order=0)
    if normalize:
        raster_arr_flt = np.abs(raster_arr_flt)
        raster_arr_flt -= np.min(raster_arr_flt)
        raster_arr_flt /= np.ptp(raster_arr_flt)
    raster_arr[np.isnan(raster_arr)] = NO_DATA_VALUE
    write_raster(raster_arr_flt, input_raster_file, transform=input_raster_file.transform,
                 outfile_path=outfile_path)


def get_raster_extents(gdal_raster):
    """
    Get Raster Extents
    :param gdal_raster: Input gdal raster object
    :return: (Xmin, YMax, Xmax, Ymin)
    """
    transform = gdal_raster.GetGeoTransform()
    ulx, uly = transform[0], transform[3]
    xres, yres = transform[1], transform[5]
    lrx, lry = ulx + xres * gdal_raster.RasterXSize, uly + yres * gdal_raster.RasterYSize
    return str(ulx), str(lry), str(lrx), str(uly)


def gdal_warp_syscall(input_raster_file, outfile_path, resampling_factor=3, resampling_func=gdal.GRA_NearestNeighbour,
                      downsampling=True, from_raster=None):
    """
    System call for mitigating GDALGetResampleFunction error at runtime
    :param input_raster_file: Input raster file
    :param outfile_path: Output file path
    :param resampling_factor: Resampling factor (default 3)
    :param resampling_func: Resampling function
    :param downsampling: Downsample raster (default True)
    :param from_raster: Reproject input raster considering another raster
    :return: None
    """

    src_raster_file = gdal.Open(input_raster_file)
    rfile = src_raster_file
    if from_raster:
        rfile = gdal.Open(from_raster)
        resampling_factor = 1
    src_band = rfile.GetRasterBand(1)
    transform = rfile.GetGeoTransform()
    xRes, yRes = transform[1], transform[5]
    extent = get_raster_extents(rfile)
    dst_proj = rfile.GetProjection()
    no_data = src_band.GetNoDataValue()
    if not downsampling:
        resampling_factor = 1 / resampling_factor
    xRes, yRes = xRes * resampling_factor, yRes * resampling_factor
    resampling_dict = {gdal.GRA_NearestNeighbour: 'near', gdal.GRA_Bilinear: 'bilinear', gdal.GRA_Cubic: 'cubic',
                       gdal.GRA_CubicSpline: 'cubicspline', gdal.GRA_Lanczos: 'lanczos', gdal.GRA_Average: 'average',
                       gdal.GRA_Mode: 'mode', gdal.GRA_Max: 'max', gdal.GRA_Min: 'min', gdal.GRA_Med: 'med',
                       gdal.GRA_Q1: 'q1', gdal.GRA_Q3: 'q3'}
    resampling_func = resampling_dict[resampling_func]
    sys_call = ['/usr/local/Cellar/gdal/2.4.2/bin/gdalwarp', '-t_srs', dst_proj, '-te', extent[0], extent[1],
                extent[2], extent[3], '-dstnodata', str(no_data), '-r', str(resampling_func), '-tr', str(xRes),
                str(yRes), '-overwrite', input_raster_file, outfile_path]
    subprocess.call(sys_call)
