# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import wget


def download_ssebop_data(sse_link, year_list, month_list, outdir):
    """
    Download SSEBop Data
    :param sse_link: Main SSEBop link without file name
    :param year_list: List of years
    :param month_list: List of months
    :param outdir: Download directory
    :return: None
    """

    for year in year_list:
        for month in month_list:
            month_str = str(month)
            if 1 <= month <= 9:
                month_str = '0' + month_str
            url = sse_link + 'm' + str(year) + month_str + '.zip'
            local_file_name = outdir + 'SSEBop_' + str(year) + month_str + '.zip'
            wget.download(url, local_file_name)


sse_link = 'https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/monthly/downloads/'
outdir = '../Data/SSEBop/'
year_list = range(2002, 2019)
month_list = range(4, 9)
download_ssebop_data(sse_link, year_list, month_list, outdir)

