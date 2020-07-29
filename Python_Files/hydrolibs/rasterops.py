# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import matplotlib.pyplot as plt
import rasterio as rio
import geopandas as gpd
import numpy as np
import gdal
import astropy.convolution as apc
import scipy.ndimage.filters as flt
import subprocess
import xmltodict
import os
import multiprocessing
from joblib import Parallel, delayed
from rasterio.plot import plotting_extent
from rasterio.mask import mask
from shapely.geometry import mapping
from collections import defaultdict
from datetime import datetime
from glob import glob
from Python_Files.hydrolibs.sysops import make_gdal_sys_call_str

NO_DATA_VALUE = -32767.0


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
        if raster_file.nodata:
            raster_arr[np.isclose(raster_arr, raster_file.nodata)] = np.nan
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


def crop_raster(input_raster_file, input_mask_path, outfile_path, plot_fig=False, plot_title="", ext_mask=True,
                gdal_path='/usr/local/Cellar/gdal/2.4.2/bin/', verbose=True):
    """
    Crop raster data based on given shapefile
    :param input_raster_file: Input raster dataset path
    :param input_mask_path: Shapefile path
    :param outfile_path: Output file path (only tiff file)
    :param plot_fig: If true, then cropped raster data is plotted
    :param plot_title: Plot title to display
    :param ext_mask: Set true to extract raster by mask file
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param verbose: Set True to print system call info 
    :return: Cropped raster dataset (if ext_mask is False)
    """

    if ext_mask:
        src_raster_file = gdal.Open(input_raster_file)
        src_band = src_raster_file.GetRasterBand(1)
        transform = src_raster_file.GetGeoTransform()
        xres, yres = transform[1], transform[5]
        no_data = src_band.GetNoDataValue()
        os_sep = input_mask_path.rfind(os.sep)
        if os_sep == -1:
            os_sep = input_mask_path.rfind('/')
        layer_name = input_mask_path[os_sep + 1: input_mask_path.rfind('.')]
        args = ['-tr', str(xres), str(yres), '-tap', '-cutline', input_mask_path, '-cl', layer_name,
                '-crop_to_cutline', '-dstnodata', str(no_data), '-overwrite', '-ot', 'Float32', '-of', 'GTiff',
                input_raster_file, outfile_path]
        sys_call = make_gdal_sys_call_str(gdal_path=gdal_path, gdal_command='gdalwarp', args=args, verbose=verbose)
        subprocess.call(sys_call)
    else:
        shape_file = gpd.read_file(input_mask_path)
        shape_file_geom = mapping(shape_file['geometry'][0])
        raster_file = rio.open(input_raster_file)
        raster_crop, raster_transform = mask(raster_file, [shape_file_geom], crop=True)
        shape_extent = plotting_extent(raster_crop[0], raster_transform)
        raster_crop = np.squeeze(raster_crop)
        write_raster(raster_crop, raster_file, transform=raster_transform, outfile_path=outfile_path,
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


def reclassify_raster2(input_raster_file, class_dict, outfile_path):
    """
    Reclassify raster data based on given class dictionary (left and right inclusive)
    :param input_raster_file: Input raster file path
    :param class_dict: Classification dictionary containing (from, to) as keys and "becomes" as value
    :param outfile_path: Output file path (only tiff file)
    :return: Reclassified raster
    """

    raster_arr, raster_file = read_raster_as_arr(input_raster_file, change_dtype=False)
    for key in class_dict.keys():
        raster_arr[np.logical_and(raster_arr >= key[0], raster_arr <= key[1])] = class_dict[key]
    raster_arr = raster_arr.astype(np.float32)
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
    for raster_file in glob(input_dir + os.sep + pattern):
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
    raster_arr[raster_arr != val] = NO_DATA_VALUE
    write_raster(raster_arr, input_raster_file, transform=input_raster_file.transform, outfile_path=outfile_path)
    return raster_arr


def fill_nans(input_raster_file, ref_file, outfile_path, fill_value=0):
    """
    Fill nan values in a raster considering a reference raster
    :param input_raster_file: Input raster file
    :param ref_file: Reference raster to consider
    :param outfile_path: Output raster path
    :param fill_value: Value to replace nans
    :return: None
    """

    raster_arr, input_raster_file = read_raster_as_arr(input_raster_file)
    ref_arr = read_raster_as_arr(ref_file, get_file=False)
    raster_arr[np.where(np.logical_and(~np.isnan(ref_arr), np.isnan(raster_arr)))] = fill_value
    write_raster(raster_arr, input_raster_file, transform=input_raster_file.transform, outfile_path=outfile_path)


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


def apply_gaussian_filter(input_raster_file, ref_file, outfile_path, sigma=3, normalize=False, ignore_nan=True):
    """
    Apply a gaussian filter over a raster image
    :param input_raster_file: Input raster file
    :param ref_file: Reference raster having continuous data for selecting appropriate AOI
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
    else:
        raster_arr[np.isnan(raster_arr)] = 0
        raster_arr_flt = flt.gaussian_filter(raster_arr, sigma=sigma, order=0)
    if normalize:
        raster_arr_flt = np.abs(raster_arr_flt)
        raster_arr_flt -= np.min(raster_arr_flt)
        raster_arr_flt /= np.ptp(raster_arr_flt)
    ref_arr = read_raster_as_arr(ref_file, get_file=False)
    raster_arr_flt[np.isnan(ref_arr)] = NO_DATA_VALUE
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


def reproject_raster(input_raster_file, outfile_path, resampling_factor=1, resampling_func=gdal.GRA_NearestNeighbour,
                     downsampling=True, from_raster=None, keep_original=False, gdal_path='/usr/bin/', verbose=True):
    """
    Reproject raster using GDAL system call
    :param input_raster_file: Input raster file
    :param outfile_path: Output file path
    :param resampling_factor: Resampling factor (default 3)
    :param resampling_func: Resampling function
    :param downsampling: Downsample raster (default True)
    :param from_raster: Reproject input raster considering another raster
    :param keep_original: Set True to only use the new projection system from 'from_raster'. The original raster extent
    is not changed
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param verbose: Set True to print system call info
    :return: None
    """

    src_raster_file = gdal.Open(input_raster_file)
    rfile = src_raster_file
    if from_raster and not keep_original:
        rfile = gdal.Open(from_raster)
        resampling_factor = 1
    src_band = rfile.GetRasterBand(1)
    transform = rfile.GetGeoTransform()
    xres, yres = transform[1], transform[5]
    extent = get_raster_extents(rfile)
    dst_proj = rfile.GetProjection()
    no_data = src_band.GetNoDataValue()
    if not downsampling:
        resampling_factor = 1 / resampling_factor
    xres, yres = xres * resampling_factor, yres * resampling_factor

    resampling_dict = {gdal.GRA_NearestNeighbour: 'near', gdal.GRA_Bilinear: 'bilinear', gdal.GRA_Cubic: 'cubic',
                       gdal.GRA_CubicSpline: 'cubicspline', gdal.GRA_Lanczos: 'lanczos', gdal.GRA_Average: 'average',
                       gdal.GRA_Mode: 'mode', gdal.GRA_Max: 'max', gdal.GRA_Min: 'min', gdal.GRA_Med: 'med',
                       gdal.GRA_Q1: 'q1', gdal.GRA_Q3: 'q3'}
    resampling_func = resampling_dict[resampling_func]
    args = ['-t_srs', dst_proj, '-te', extent[0], extent[1], extent[2], extent[3],
            '-dstnodata', str(no_data), '-r', str(resampling_func), '-tr', str(xres), str(yres), '-ot', 'Float32',
            '-overwrite', input_raster_file, outfile_path]
    sys_call = make_gdal_sys_call_str(gdal_path=gdal_path, gdal_command='gdalwarp', args=args, verbose=verbose)
    subprocess.call(sys_call)


def crop_rasters(input_raster_dir, input_mask_file, outdir, pattern='*.tif', ext_mask=True,
                 gdal_path='/usr/local/Cellar/gdal/2.4.2/bin/', verbose=True):
    """
    Crop multiple rasters in a directory
    :param input_raster_dir: Directory containing raster files which are named as *_<Year>.*
    :param input_mask_file: Mask file (shapefile) used for cropping
    :param outdir: Output directory for storing masked rasters
    :param pattern: Raster extension
    :param ext_mask: Set False to extract by geometry only
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param verbose: Set True to print system call info
    :return: None
    """

    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1:]
        crop_raster(raster_file, input_mask_file, out_raster, ext_mask=ext_mask, gdal_path=gdal_path, verbose=verbose)


def smooth_rasters(input_raster_dir, ref_file, outdir, pattern='*_Masked.tif', sigma=5, normalize=False,
                   ignore_nan=False):
    """
    Smooth rasters using Gaussian Filter
    :param input_raster_dir:  Directory containing raster files which are named as *_<Year>.*
    :param ref_file: Reference raster for discarding nans
    :param outdir: Output directory for storing masked rasters
    :param pattern: Raster extension
    :param sigma: Standard Deviation for Gaussian Filter
    :param normalize: Set true to normalize the filered values
    :param ignore_nan: Set True to use astropy convolution
    :return: None
    """

    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1: raster_file.rfind('.')] + '_Smoothed.tif'
        apply_gaussian_filter(raster_file, ref_file=ref_file, outfile_path=out_raster, sigma=sigma, normalize=normalize,
                              ignore_nan=ignore_nan)


def reproject_rasters(input_raster_dir, ref_raster, outdir, pattern='*.tif', gdal_path='/usr/bin/', verbose=True):
    """
    Reproject rasters in a directory
    :param input_raster_dir: Directory containing raster files which are named as *_<Year>.*
    :param ref_raster: Reference raster file to consider while reprojecting
    :param outdir: Output directory for storing reprojected rasters
    :param pattern: Raster extension
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param verbose: Set True to print system call info
    :return: None
    """

    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1:]
        reproject_raster(raster_file, from_raster=ref_raster, outfile_path=out_raster, gdal_path=gdal_path, 
                         verbose=verbose)


def mask_rasters(input_raster_dir, ref_raster, outdir, pattern='*.tif'):
    """
    Mask out a raster using another raster
    :param input_raster_dir: Directory containing raster files which are named as *_<Year>.*
    :param ref_raster: Reference raster file to consider while masking
    :param outdir: Output directory for storing reprojected rasters
    :param pattern: Raster extension
    :return: None
    """

    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1:]
        filter_nans(raster_file, ref_raster, outfile_path=out_raster)


def apply_et_filter(input_raster_dir, ref_raster1, ref_raster2, outdir, pattern='ET_*.tif', flt_values=(1,)):
    """
    Mask out a raster using another raster
    :param input_raster_dir: Directory containing raster files which are named as *_<Year>.*
    :param ref_raster1: Reference raster file to consider while masking
    :param ref_raster2: Filter out nan values using this raster
    :param outdir: Output directory for storing reprojected rasters
    :param pattern: Raster extension
    :param flt_values: Tuple of filter values
    :return: None
    """

    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1: raster_file.rfind('.')] + '_flt.tif'
        apply_raster_filter(ref_raster1, raster_file, outfile_path=out_raster, flt_values=flt_values)
        filter_nans(out_raster, ref_file=ref_raster2, outfile_path=out_raster)


def retrieve_pixel_coords(geo_coord, data_source, gdal_path='/usr/local/Cellar/gdal/2.4.2/bin/', verbose=True):
    """
    Get pixels coordinates from geo-coordinates
    :param geo_coord: Geo-cooridnate tuple
    :param data_source: Original GDAL reference having affine transformation parameters
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :return: Pixel coordinates in x and y direction (should be reversed in the caller function to get the actual pixel
    :param verbose: Set True to print system call info
    position)
    """

    args = ['-xml', '-geoloc', data_source, str(geo_coord[0]), str(geo_coord[1])]
    sys_call = make_gdal_sys_call_str(gdal_path=gdal_path, gdal_command='gdallocationinfo', args=args, verbose=verbose)
    p = subprocess.Popen(sys_call, stdout=subprocess.PIPE)
    p.wait()
    gdalloc_xml = xmltodict.parse(p.stdout.read())
    px, py = int(gdalloc_xml['Report']['@pixel']), int(gdalloc_xml['Report']['@line'])
    return px, py


def compute_raster_shp(input_raster_file, input_shp_file, outfile_path, nan_fill=0, point_arithmetic='sum',
                       value_field_pos=0, gdal_path='/usr/local/Cellar/gdal/2.4.2/bin/', verbose=True):
    """
    Replace/Insert values in an existing raster based on the point coordinates from the shape file and applying suitable
    arithmetic on the point values (the raster and the shape file must be having the same CRS)
    :param input_raster_file: Input raster file
    :param input_shp_file: Input shape file (point layer only)
    :param outfile_path: Output raster file path
    :param nan_fill: This value is for filling up raster cells where there are no points present from the shapefile
    :param point_arithmetic: Apply sum operation cummulatively on the point values (use None' to keep as is
    or use 'mean' for using the mean of the point values within a particular raster pixel)
    :param value_field_pos: Shapefile value field position to use (zero indexing)
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param verbose: Set True to print system call info
    :return: None
    """

    shp_data = gpd.read_file(input_shp_file)
    raster_arr, raster_file = read_raster_as_arr(input_raster_file)
    raster_arr[~np.isnan(raster_arr)] = nan_fill
    count_arr = np.full_like(raster_arr, fill_value=0)
    count_arr[np.isnan(raster_arr)] = np.nan
    maxy, maxx = raster_arr.shape
    for idx, point in np.ndenumerate(shp_data['geometry']):
        geocoords = point.x, point.y
        px, py = retrieve_pixel_coords(geocoords, input_raster_file, gdal_path=gdal_path, verbose=verbose)
        pval = shp_data[shp_data.columns[value_field_pos]][idx[0]]
        if py < maxy and px < maxx:
            if np.isnan(raster_arr[py, px]):
                raster_arr[py, px] = 0
            if point_arithmetic == 'sum':
                raster_arr[py, px] += pval
                count_arr[py, px] += 1
            elif point_arithmetic == 'None':
                raster_arr[py, px] = pval
    if point_arithmetic == 'mean':
        raster_arr = raster_arr / count_arr
    raster_arr[np.isnan(raster_arr)] = NO_DATA_VALUE
    write_raster(raster_arr, raster_file, transform=raster_file.transform, outfile_path=outfile_path)


def compute_rasters_from_shp(input_raster_dir, input_shp_dir, outdir, nan_fill=0, point_arithmetic='sum',
                             value_field_pos=0, pattern='*.tif', gdal_path='/usr/local/Cellar/gdal/2.4.2/bin/', 
                             verbose=True):
    """
    Replace/Insert values of all rasters in a directory based on the point coordinates from the shape file and applying
    suitable arithmetic on the point values
    :param input_raster_dir: Input raster directory
    :param input_shp_dir: Input shape file directory (point layer only)
    :param outdir: Output raster directory
    :param nan_fill: This value is for filling up raster cells where there are no points present from the shapefile
    :param point_arithmetic: Apply sum operation cummulatively on the point values (use None' to keep as is
    or use 'mean' for using the mean of the point values within a particular raster pixel)
    :param value_field_pos: Shapefile value field position to use (zero indexing)
    :param pattern: Raster extension
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param verbose: Set True to print system call info
    :return: None
    """

    raster_files, shp_files = glob(input_raster_dir + pattern), glob(input_shp_dir + '*.shp')
    raster_files.sort()
    shp_files.sort()
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(parellel_raster_compute)(raster_file, shp_file, outdir=outdir, nan_fill=nan_fill,
                                                                point_arithmetic=point_arithmetic,
                                                                value_field_pos=value_field_pos, gdal_path=gdal_path,
                                                                verbose=verbose)
                               for raster_file, shp_file in zip(raster_files, shp_files))


def parellel_raster_compute(raster_file, shp_file, outdir, nan_fill=0, point_arithmetic='sum', value_field_pos=0,
                            gdal_path='/usr/local/Cellar/gdal/2.4.2/bin/', verbose=True):
    """
    Use this from #compute_rasters_shp to parallelize raster creation from shpfiles
    :param raster_file: Input raster file
    :param shp_file: Input shape file
    :param outdir: Output raster directory
    :param nan_fill: This value is for filling up raster cells where there are no points present from the shapefile
    :param point_arithmetic: Apply sum operation cummulatively on the point values (use None' to keep as is
    or use 'mean' for using the mean of the point values within a particular raster pixel)
    :param value_field_pos: Shapefile value field position to use (zero indexing)
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param verbose: Set True to print system call info
    :return: None
    """

    out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1:]
    print('\nProcessing for', raster_file, shp_file, '...')
    compute_raster_shp(raster_file, input_shp_file=shp_file, outfile_path=out_raster, nan_fill=nan_fill,
                       point_arithmetic=point_arithmetic, value_field_pos=value_field_pos, gdal_path=gdal_path,
                       verbose=verbose)


def convert_gw_data(input_raster_dir, outdir, pattern='*.tif'):
    """
    Convert groundwater data (in acreft) to mm
    :param input_raster_dir: Input raster directory
    :param outdir: Output raster directory
    :param pattern: Raster extension
    :return: None
    """

    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1:]
        raster_arr, raster_ref = read_raster_as_arr(raster_file)
        transform = raster_ref.get_transform()
        xres, yres = transform[1] / 1000., transform[5] / 1000.
        raster_arr[~np.isnan(raster_arr)] *= 1.233 / (np.abs(xres * yres))
        raster_arr[np.isnan(raster_arr)] = NO_DATA_VALUE
        write_raster(raster_arr, raster_ref, transform=raster_ref.transform, outfile_path=out_raster)


def scale_raster_data(input_raster_dir, outdir, scaling_factor=10, pattern='*.tif'):
    """
    Scale raster data if required
    :param input_raster_dir: Input raster directory
    :param outdir: Output raster directory
    :param scaling_factor: Scaling factor
    :param pattern: Raster extension
    :return: None
    """

    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1:]
        raster_arr, raster_ref = read_raster_as_arr(raster_file)
        raster_arr[~np.isnan(raster_arr)] *= scaling_factor
        raster_arr[np.isnan(raster_arr)] = NO_DATA_VALUE
        write_raster(raster_arr, raster_ref, transform=raster_ref.transform, outfile_path=out_raster)


def fill_mean_value(input_raster_dir, outdir, pattern='GRACE*.tif'):
    """
    Replace all values with the mean value of the raster
    :param input_raster_dir: Input raster directory
    :param outdir: Output directory
    :param pattern: Raster file name pattern
    :return: None
    """

    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1:]
        raster_arr, raster_file = read_raster_as_arr(raster_file)
        mean_val = np.nanmean(raster_arr)
        raster_arr[np.isnan(raster_arr)] = NO_DATA_VALUE
        raster_arr[raster_arr != NO_DATA_VALUE] = mean_val
        write_raster(raster_arr, raster_file, transform=raster_file.transform, outfile_path=out_raster)


def create_raster_dict(input_raster_dir, pattern='*.tif'):
    """
    Create a raster dictionary keyed by years
    :param input_raster_dir: Input raster directory
    :param pattern: File pattern
    :return: Dictionary of rasters present in the directory
    """

    raster_dict = {}
    for raster_file in glob(input_raster_dir + pattern):
        year = raster_file[raster_file.rfind('_') + 1: raster_file.rfind('.')]
        raster_dict[int(year)] = read_raster_as_arr(raster_file, get_file=False)
    return raster_dict


def create_yearly_avg_raster_dict(input_raster_dir, pattern='GRACE*.tif'):
    """
    Create a raster dictionary keyed by years and the values averaged over each year
    :param input_raster_dir: Input raster directory
    :param pattern: File pattern
    :return: Dictionary of rasters present in the directory
    """

    raster_dict = defaultdict(lambda: [])
    for raster_file in glob(input_raster_dir + pattern):
        year = raster_file[raster_file.rfind('_') + 1: raster_file.rfind('.')]
        raster_arr = read_raster_as_arr(raster_file, get_file=False)
        raster_dict[int(year)].append(raster_arr)
    yearly_avg_raster_dict = {}
    for year in raster_dict.keys():
        raster_list = raster_dict[year]
        sum_arr = np.full_like(raster_list[0], fill_value=0)
        for raster in raster_list:
            sum_arr += raster
        sum_arr /= len(raster_list)
        yearly_avg_raster_dict[year] = np.nanmean(sum_arr)
    return yearly_avg_raster_dict


def create_monthly_avg_raster_dict(input_raster_dir, pattern='GRACE*.tif'):
    """
    Create a raster dictionary keyed by years and the values averaged over each year
    :param input_raster_dir: Input raster directory
    :param pattern: File pattern
    :return: Dictionary of rasters present in the directory
    """

    raster_dict = {}
    for raster_file in glob(input_raster_dir + pattern):
        file_name = raster_file[raster_file.rfind(os.sep) + 1:]
        dt = file_name[file_name.find('_') + 1: file_name.rfind('.')]
        dt = datetime.strptime(dt, '%b_%Y')
        raster_arr = read_raster_as_arr(raster_file, get_file=False)
        raster_dict[dt] = np.nanmean(raster_arr)
    return raster_dict


def fix_large_values(input_raster_dir, outdir, max_threshold=1e+5, pattern='GW*.tif'):
    """
    Fix unusually large values introduced by gdal_rasterize sometimes
    :param input_raster_dir: Input raster directory
    :param outdir: Output directory
    :param max_threshold: Max value beyond which values will be set to no data value, default unit is acrefit
    :param pattern: File pattern
    :return: None
    """

    for raster_file in glob(input_raster_dir + pattern):
        out_raster = outdir + raster_file[raster_file.rfind(os.sep) + 1:]
        raster_arr, raster_file = read_raster_as_arr(raster_file)
        raster_arr[np.isnan(raster_arr)] = NO_DATA_VALUE
        raster_arr[raster_arr >= max_threshold] = NO_DATA_VALUE
        write_raster(raster_arr, raster_file, transform=raster_file.transform, outfile_path=out_raster)


def generate_ssebop_raster_list(ssebop_dir, year, month_list):
    """
    Generate SSEBop raster list based on a list of months
    :param ssebop_dir: Input SSEBOp directory
    :param year: SSEBop year in %Y format
    :param month_list: List of months in %m format
    :return: Raster array list and raster reference list as a tuple
    """

    ssebop_raster_arr_list = []
    ssebop_raster_file_list = []
    for month in month_list:
        month_str = str(month)
        if 1 <= month <= 9:
            month_str = '0' + month_str
        pattern = '*' + str(year) + month_str + '*.tif'
        ssebop_file = glob(ssebop_dir + pattern)[0]
        ssebop_raster_arr, ssebop_raster_file = read_raster_as_arr(ssebop_file)
        ssebop_raster_arr_list.append(ssebop_raster_arr)
        ssebop_raster_file_list.append(ssebop_raster_file)
    return ssebop_raster_arr_list, ssebop_raster_file_list


def generate_cummulative_ssebop(ssebop_dir, year_list, start_month, end_month, out_dir):
    """
    Generate cummulative SSEBop data
    :param ssebop_dir: SSEBop directory
    :param year_list: List of years
    :param start_month: Start month in %m format
    :param end_month: End month in %m format
    :param out_dir: Output directory
    :return: None
    """

    month_flag = False
    month_list = []
    actual_start_year = year_list[0]
    if end_month <= start_month:
        year_list = [actual_start_year - 1] + list(year_list)
        month_flag = True
    else:
        month_list = range(start_month, end_month + 1)
    for year in year_list:
        actual_year = year
        if month_flag:
            actual_year += 1
        print('Generating cummulative SSEBop for', actual_year, '...')
        if month_flag:
            month_list_y1 = range(start_month, 13)
            month_list_y2 = range(1, end_month + 1)
            ssebop_raster_arr_list1, ssebop_raster_file_list1 = generate_ssebop_raster_list(ssebop_dir, year,
                                                                                            month_list_y1)
            ssebop_raster_arr_list2, ssebop_raster_file_list2 = generate_ssebop_raster_list(ssebop_dir, year + 1,
                                                                                            month_list_y2)
            ssebop_raster_arr_list = ssebop_raster_arr_list1 + ssebop_raster_arr_list2
            ssebop_raster_file_list = ssebop_raster_file_list1 + ssebop_raster_file_list2
        else:
            ssebop_raster_arr_list, ssebop_raster_file_list = generate_ssebop_raster_list(ssebop_dir, year, month_list)
        sum_arr_ssebop = ssebop_raster_arr_list[0]
        for ssebop_raster_arr in ssebop_raster_arr_list[1:]:
            sum_arr_ssebop += ssebop_raster_arr
        sum_arr_ssebop[np.isnan(sum_arr_ssebop)] = NO_DATA_VALUE
        ssebop_raster_file = ssebop_raster_file_list[0]
        out_ssebop = out_dir + 'SSEBop_' + str(year) + '.tif'
        if month_flag:
            out_ssebop = out_dir + 'SSEBop_' + str(actual_year) + '.tif'
        write_raster(sum_arr_ssebop, ssebop_raster_file, transform=ssebop_raster_file.transform,
                     outfile_path=out_ssebop)
        if month_flag and year == year_list[-1] - 1:
            return


def create_crop_coeff_raster(input_cdl_file, output_file):
    """
    Create crop coefficient raster based on NASS CDL file
    :param input_cdl_file: Input CDL file path
    :param output_file: Output file path
    :return: None
    """

    cdl_arr, cdl_file = read_raster_as_arr(input_cdl_file)
    crop_coeff_arr = np.full_like(cdl_arr, fill_value=1.0)
    crop_coeff_dict = {1: 1.2, 2: 1.2, 3: 1.2, 4: 1.15, 5: 1.15, 6: 1.15, 10: 1.15, 12: 1.15, 13: 1.2, 14: 1.15,
                       21: 1.15, 22: 1.15, 23: 1.15, 24: 1.15, 25: 1.2, 26: 1.15, 27: 1.05, 28: 1.15, 29: 1.0, 30: 1.15,
                       31: 1.15, 32: 1.10, 33: 1.15, 34: 1.15, 36: 1.2, 37: 1.0, 39: 1.15, 41: 1.20, 42: 1.15, 43: 1.15,
                       45: 1.25, 46: 1.15, 48: 1.0, 49: 1.05, 50: 1.0, 51: 1.0, 52: 1.10, 53: 1.15, 54: 1.15, 55: 1.05,
                       56: 1.05, 58: 1.15, 66: 1.20, 67: 1.15, 68: 1.15, 69: 0.85, 72: 0.70, 74: 1.15, 75: 0.90,
                       76: 1.10, 77: 1.20, 204: 1.10, 206: 1.05, 207: 0.95, 208: 1.0, 209: 0.85, 211: 0.70, 212: 0.70,
                       214: 1.05, 215: 0.85, 216: 1.05, 220: 1.15, 221: 0.85, 222: 0.95, 223: 1.15, 225: 1.15,
                       226: 1.15, 227: 1.0, 228: 1.0, 230: 2.15, 231: 1.85, 232: 2.20, 233: 2.15, 234: 2.35, 235: 2.35,
                       236: 2.35, 237: 2.35, 238: 2.35, 239: 2.35, 240: 2.30, 241: 2.35, 242: 1.05, 243: 1.05,
                       244: 1.05, 245: 1.05, 246: 0.90, 247: 1.10, 248: 1.05, 250: 1.05, 254: 2.30}
    for crop_code in crop_coeff_dict.keys():
        crop_coeff_arr[cdl_arr == crop_code] = crop_coeff_dict[crop_code]
    crop_coeff_arr[cdl_arr == 0] = NO_DATA_VALUE
    write_raster(crop_coeff_arr, cdl_file, transform=cdl_file.transform, outfile_path=output_file)


def update_crop_coeff_raster(input_crop_coeff_raster, agri_raster):
    """
    Update crop coefficient raster based on AGRI raster
    :param input_crop_coeff_raster: Input crop coefficient raster file path
    :param agri_raster: Input AGRI raster file path
    :return: None
    """

    crop_coeff_arr, crop_coeff_file = read_raster_as_arr(input_crop_coeff_raster)
    agri_arr = read_raster_as_arr(agri_raster, get_file=False)
    crop_coeff_arr[np.logical_and(crop_coeff_arr == 1., agri_arr < 0.1)] = 0.
    write_raster(crop_coeff_arr, crop_coeff_file, transform=crop_coeff_file.transform,
                 outfile_path=input_crop_coeff_raster)
