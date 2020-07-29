# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import ee
import requests
import zipfile
import os
import xmltodict
import geopandas as gpd
from glob import glob


def download_gee_data(year_list, start_month, end_month, aoi_shp_file, outdir):
    """
    Download MOD16 and PRISM data. MOD16 has to be divided by 10 (line 38) as its original scale is 0.1 mm/8 days.
    :param year_list: List of years in %Y format
    :param start_month: Start month in %m format
    :param end_month: End month in %m format
    :param aoi_shp_file: Area of interest shapefile (must be in WGS84)
    :param outdir: Download directory
    :return: None
    """

    ee.Initialize()
    mod16_collection = ee.ImageCollection("MODIS/006/MOD16A2")
    prism_collection = ee.ImageCollection("OREGONSTATE/PRISM/AN81m")
    aoi_shp = gpd.read_file(aoi_shp_file)
    minx, miny, maxx, maxy = aoi_shp.geometry.total_bounds
    gee_aoi = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    for year in year_list:
        start_date = ee.Date.fromYMD(year, start_month, 1)
        if end_month == 12:
            end_date = ee.Date.fromYMD(year + 1, 1, 1)
        else:
            end_date = ee.Date.fromYMD(year, end_month + 1, 1)
        if end_month <= start_month:
            start_date = ee.Date.fromYMD(year - 1, start_month, 1)
        mod16_total = mod16_collection.select('ET').filterDate(start_date, end_date).sum().divide(10).toDouble()
        prism_total = prism_collection.select('ppt').filterDate(start_date, end_date).sum().toDouble()
        mod16_url = mod16_total.getDownloadUrl({
            'scale': 1000,
            'crs': 'EPSG:4326',
            'region': gee_aoi
        })
        prism_url = prism_total.getDownloadUrl({
            'scale': 1000,
            'crs': 'EPSG:4326',
            'region': gee_aoi
        })
        gee_vars = ['ET_', 'P_']
        gee_links = [mod16_url, prism_url]
        for gee_var, gee_url in zip(gee_vars, gee_links):
            local_file_name = outdir + gee_var + str(year) + '.zip'
            print('Dowloading', local_file_name, '...')
            r = requests.get(gee_url, allow_redirects=True)
            open(local_file_name, 'wb').write(r.content)


def download_ssebop_data(sse_link, year_list, start_month, end_month, outdir):
    """
    Download SSEBop Data
    :param sse_link: Main SSEBop link without file name
    :param year_list: List of years
    :param start_month: Start month in %m format
    :param end_month: End month in %m format
    :param outdir: Download directory
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
        print('Downloading SSEBop for', year, '...')
        if month_flag:
            month_list = list(range(start_month, 13))
            if actual_start_year <= year < year_list[-1]:
                month_list = list(range(1, end_month + 1)) + month_list
            elif year == year_list[-1]:
                month_list = list(range(1, end_month + 1))
        for month in month_list:
            month_str = str(month)
            if 1 <= month <= 9:
                month_str = '0' + month_str
            url = sse_link + 'm' + str(year) + month_str + '.zip'
            local_file_name = outdir + 'SSEBop_' + str(year) + month_str + '.zip'
            r = requests.get(url, allow_redirects=True)
            open(local_file_name, 'wb').write(r.content)


def download_cropland_data(aoi_shp_file, outfile, year=2015):
    """
    Download USDA-NASS cropland data
    :param aoi_shp_file: Area of interest shapefile
    :param outfile: Output file name
    :param year: Year in %Y format
    :return: None
    """

    print('Downloading CDL data for', str(year), '...')
    nass_proj_wkt = 'PROJCS["NAD_1983_Albers",' \
                    'GEOGCS["NAD83",' \
                    'DATUM["North_American_Datum_1983",' \
                    'SPHEROID["GRS 1980",6378137,298.257222101,' \
                    'AUTHORITY["EPSG","7019"]],' \
                    'TOWGS84[0,0,0,0,0,0,0],' \
                    'AUTHORITY["EPSG","6269"]],' \
                    'PRIMEM["Greenwich",0,' \
                    'AUTHORITY["EPSG","8901"]],' \
                    'UNIT["degree",0.0174532925199433,' \
                    'AUTHORITY["EPSG","9108"]],' \
                    'AUTHORITY["EPSG","4269"]],' \
                    'PROJECTION["Albers_Conic_Equal_Area"],' \
                    'PARAMETER["standard_parallel_1",29.5],' \
                    'PARAMETER["standard_parallel_2",45.5],' \
                    'PARAMETER["latitude_of_center",23],' \
                    'PARAMETER["longitude_of_center",-96],' \
                    'PARAMETER["false_easting",0],' \
                    'PARAMETER["false_northing",0],' \
                    'UNIT["meters",1]]'
    aoi_shp = gpd.read_file(aoi_shp_file)
    aoi_shp = aoi_shp.to_crs(nass_proj_wkt)
    minx, miny, maxx, maxy = aoi_shp.geometry.total_bounds
    nass_xml_url = ' https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile?year=' + str(year) + '&bbox=' + \
                   str(minx) + ',' + str(miny) + ',' + str(maxx) + ',' + str(maxy)
    r = requests.get(nass_xml_url, allow_redirects=True)
    nass_data_url = xmltodict.parse(r.content)['ns1:GetCDLFileResponse']['returnURL']
    r = requests.get(nass_data_url, allow_redirects=True)
    open(outfile, 'wb').write(r.content)


def extract_data(zip_dir, out_dir, rename_extracted_files=False):
    """
    Extract data from zip file
    :param zip_dir: Input zip directory
    :param out_dir: Output directory to write extracted files
    :param rename_extracted_files: Set True to rename extracted files according the original zip file name
    :return: None
    """

    print('Extracting zip files...')
    for zip_file in glob(zip_dir + '*.zip'):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            if rename_extracted_files:
                zip_info = zip_ref.infolist()[0]
                zip_info.filename = zip_file[zip_file.rfind(os.sep) + 1: zip_file.rfind('.')] + '.tif'
                zip_ref.extract(zip_info, path=out_dir)
            else:
                zip_ref.extractall(path=out_dir)
