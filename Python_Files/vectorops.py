# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import rasterio as rio
import numpy as np
import subprocess
from glob import glob


def reproject_vector(input_vector_file, outfile_path, ref_file, crs='epsg:4326', crs_from_file=True, raster=True):
    """
    Reproject a vector file
    :param input_vector_file: Input vector file path
    :param outfile_path: Output vector file path
    :param crs: Target CRS
    :param ref_file: Reference file (raster or vector) for obtaining target CRS
    :param crs_from_file: If true (default) read CRS from file (raster or vector)
    :param raster: If true (default) read CRS from raster else vector
    :return: Reprojected vector file in GeoPandas format
    """

    input_vector_file = gpd.read_file(input_vector_file)
    if crs_from_file:
        if raster:
            ref_file = rio.open(ref_file)
        else:
            ref_file = gpd.read_file(ref_file)
        crs = ref_file.crs
    else:
        crs = {'init': crs}
    output_vector_file = input_vector_file.to_crs(crs)
    output_vector_file.to_file(outfile_path)
    return output_vector_file


def csv2shp(input_csv_file, outfile_path, delim=',', source_crs='epsg:4326', target_crs='epsg:4326',
            long_lat_pos=(7, 8)):
    """
    Convert CSV to Shapefile
    :param input_csv_file: Input CSV file path
    :param outfile_path: Output file path
    :param delim: CSV file delimeter
    :param source_crs: CRS of the source file
    :param target_crs: Target CRS
    :param long_lat_pos: Tuple containing positions of longitude and latitude columns respectively (zero indexing)
    :return: None
    """

    input_df = pd.read_csv(input_csv_file, delimiter=delim)
    input_df = input_df.dropna(axis=1)
    long, lat = input_df.columns[long_lat_pos[0]], input_df.columns[long_lat_pos[1]]
    geometry = [Point(xy) for xy in zip(input_df[long], input_df[lat])]
    crs = {'init': source_crs}
    gdf = gpd.GeoDataFrame(input_df, crs=crs, geometry=geometry)
    gdf.to_file(outfile_path)
    if target_crs != source_crs:
        reproject_vector(outfile_path, outfile_path=outfile_path, crs=target_crs, crs_from_file=False, ref_file=None)


def shp2raster(input_shp_file, outfile_path, value_field_pos=2, xres=1000., yres=1000.,
               gdal_rasterize_path='/usr/local/Cellar/gdal/2.4.2/bin/gdal_rasterize'):
    """
    Convert Shapefile to Raster TIFF file
    :param input_shp_file: Input Shapefile path
    :param outfile_path: Output TIFF file path
    :param value_field_pos: Value field position (zero indexing)
    :param xres: Pixel width in geographic units
    :param yres: Pixel height in geographic units
    :param gdal_rasterize_path: Path to gdal_rasterize binary
    :return: None
    """

    ext_pos = input_shp_file.rfind('.')
    layer_name = input_shp_file[input_shp_file.rfind('/') + 1: ext_pos]
    shp_file = gpd.read_file(input_shp_file)
    value_field = shp_file.columns[value_field_pos]
    minx, miny, maxx, maxy = shp_file.geometry.total_bounds
    no_data_value = np.finfo(np.float32).min
    sys_call = [gdal_rasterize_path, '-l', layer_name, '-a', value_field, '-tr', str(xres), str(yres), '-te', str(minx),
                str(miny), str(maxx), str(maxy), '-ot', 'Float32', '-of', 'GTiff', '-a_nodata', str(no_data_value),
                input_shp_file, outfile_path]
    subprocess.call(sys_call)


def csvs2shps(input_dir, output_dir, pattern='*.csv', target_crs='EPSG:4326', delim=',',
              long_lat_pos=(7, 8)):
    """
    Convert all CSV files present in a folder to corresponding Shapefiles
    :param input_dir: Input directory containing csv files which are named as <Layer_Name>_<Year>.[csv|txt]
    :param output_dir: Output directory
    :param pattern: CSV  file pattern
    :param target_crs: Target CRS
    :param delim: CSV file delimeter
    :param long_lat_pos: Tuple containing positions of longitude and latitude columns respectively (zero indexing)
    :return: None
    """

    for file in glob(input_dir + pattern):
        outfile_path = output_dir + file[file.rfind('/') + 1: file.rfind('.') + 1] + 'shp'
        csv2shp(file, outfile_path=outfile_path, delim=delim, target_crs=target_crs, long_lat_pos=long_lat_pos)


def shps2rasters(input_dir, output_dir, value_field_pos=2, xres=1000, yres=1000,
                 gdal_rasterize_path='/usr/local/Cellar/gdal/2.4.2/bin/gdal_rasterize'):
    """
    Convert all Shapefiles to corresponding TIFF files
    :param input_dir: Input directory containing Shapefiles which are named as <Layer_Name>_<Year>.shp
    :param output_dir: Output directory
    :param value_field_pos: Value field position (zero indexing)
    :param xres: Pixel width in geographic units
    :param yres: Pixel height in geographic units
    :param gdal_rasterize_path: Path to gdal_rasterize binary
    :return: None
    """

    for file in glob(input_dir + '*.shp'):
        outfile_path = output_dir + file[file.rfind('/') + 1: file.rfind('.') + 1] + 'tif'
        shp2raster(file, outfile_path=outfile_path, value_field_pos=value_field_pos, xres=xres, yres=yres,
                   gdal_rasterize_path=gdal_rasterize_path)
