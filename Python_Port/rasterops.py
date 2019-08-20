# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import matplotlib.pyplot as plt
import rasterio as rio
from rasterio.plot import plotting_extent
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd
import numpy as np
# import seaborn as sns
# plt.ion()


def write_raster(raster_data, raster_file, outfile_path, bands=1):
    """
    Write raster file in GeoTIFF format
    :param raster_data: Raster data to be written
    :param raster_file: Original raster file containing geo-coordinates
    :param outfile_path: Outfile file path
    :param bands: Number of bands to write
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
            transform=raster_file.transform,
            count=bands,
            nodata=0
    ) as dst:
        dst.write(raster_data, bands)


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
    raster_data = rio.open(input_raster_file)
    raster_crop, raster_affine = mask(raster_data, [shape_file_geom])
    shape_extent = plotting_extent(raster_crop[0], raster_affine)
    raster_crop = np.squeeze(raster_crop)
    write_raster(raster_crop, raster_data, outfile_path=outfile_path)
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
    write_raster(raster_data, raster_file, outfile_path)
    return raster_data

