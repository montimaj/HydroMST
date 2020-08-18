# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import numpy as np
import pandas as pd
import os
import seaborn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sklearn.metrics as metrics
from glob import glob
from Python_Files.hydrolibs import rasterops as rops
from Python_Files.hydrolibs import vectorops as vops
from Python_Files.hydrolibs.sysops import makedirs, make_proper_dir_name
from datetime import datetime
from sklearn.linear_model import LinearRegression


def create_gw_time_series(actual_gw_file_dir, pred_gw_file_dir, grace_dir, actual_gw_pattern='GW*.tif',
                          pred_gw_pattern='pred*.tif', grace_pattern='GRACE*.tif', make_trend=False,
                          out_dir='../Outputs/'):
    """
    Create time series data for actual and predicted GW values (annual mean) along with GRACE
    :param actual_gw_file_dir: Actual GW pumping raster directory
    :param pred_gw_file_dir: Predicted GW pumping raster directory
    :param grace_dir: GRACE raster directory
    :param actual_gw_pattern: Actual GW pumping raster file pattern
    :param pred_gw_pattern: Predicted GW pumping raster file pattern
    :param grace_pattern: GRACE raster file pattern
    :param make_trend: Make trend data for the monthly grace values
    :param out_dir: Output directory for storing the CSV files
    :return: Dataframe containing year and corresponding mean values
    """

    actual_gw_raster_dict = rops.create_raster_dict(actual_gw_file_dir, pattern=actual_gw_pattern)
    pred_gw_raster_dict = rops.create_raster_dict(pred_gw_file_dir, pattern=pred_gw_pattern)
    grace_yearly_raster_dict = rops.create_yearly_avg_raster_dict(grace_dir, pattern=grace_pattern)
    grace_monthly_raster_dict = rops.create_monthly_avg_raster_dict(grace_dir, pattern=grace_pattern)
    years = sorted(list(pred_gw_raster_dict.keys()))
    dt_list = sorted(list(grace_monthly_raster_dict.keys()))
    mean_actual_gw = {}
    mean_pred_gw = {}
    mean_grace = {}
    mean_monthly_grace = {}
    for year in years:
        mean_actual_gw[year] = np.nanmean(actual_gw_raster_dict[year])
        mean_pred_gw[year] = np.nanmean(pred_gw_raster_dict[year])
        mean_grace[year] = np.nanmean(grace_yearly_raster_dict[year])
    for dt in dt_list:
        mean_monthly_grace[dt] = grace_monthly_raster_dict[dt]

    df1 = {'YEAR': years, 'Actual_GW': list(mean_actual_gw.values()), 'Pred_GW': list(mean_pred_gw.values()),
           'GRACE': list(mean_grace.values())}
    df1 = pd.DataFrame(data=df1)
    df1.to_csv(out_dir + 'ts_yearly.csv', index=False)
    time = [datetime.strftime(t, '%b %Y') for t in dt_list]
    grace_monthly_list = list(mean_monthly_grace.values())
    df2 = {'Time': time, 'GRACE': grace_monthly_list}
    if make_trend:
        grace_fit = get_trend(dt_list, grace_monthly_list)
        df2['Trend'] = grace_fit
    df2 = pd.DataFrame(data=df2)
    df2.to_csv(out_dir + 'grace_monthly.csv', index=False)
    return df1, df2


def create_gw_forecast_time_series(actual_gw_file_dir_list, pred_gw_file_dir_list, grace_csv, gmd_name_list=None,
                                   use_gmds=True, actual_gw_pattern='GW*.tif', pred_gw_pattern='pred*.tif',
                                   out_dir='../Outputs/', get_only_grace_df=False):
    """
    Create GW and GRACE dataframes
    :param actual_gw_file_dir_list: Actual GW pumping raster directory list
    :param pred_gw_file_dir_list: Predicted GW pumping raster directory list
    :param grace_csv: GRACE TWS CSV file
    :param gmd_name_list: List of GMD names
    :param use_gmds: Set False to use entire GW raster for analysis
    :param actual_gw_pattern: Actual GW pumping raster file pattern
    :param pred_gw_pattern: Predicted GW pumping raster file pattern
    :param out_dir: Output directory for storing the CSV files
    :param get_only_grace_df: Set True to get only GRACE dataframe
    :return: GRACE dataframe if get_only_grace_df=True, otherwise, tuple of three dataframes, one with the mean GW
    pumping, another with the pixelwise GW pumping values, and the other containing the monthly GRACE values
    """

    grace_df = pd.read_csv(grace_csv)
    grace_df = grace_df.dropna(axis=0)
    grace_df['GRACE'] = grace_df['GRACE'] * 10
    grace_df['DT'] = pd.to_datetime(grace_df['DT']).dt.date
    if get_only_grace_df:
        return grace_df
    gw_df = pd.DataFrame()
    gw_raster_df = pd.DataFrame()
    for index, (actual_gw_file_dir, pred_gw_file_dir) in enumerate(zip(actual_gw_file_dir_list, pred_gw_file_dir_list)):
        actual_gw_raster_dict = rops.create_raster_dict(actual_gw_file_dir, pattern=actual_gw_pattern)
        pred_gw_raster_dict = rops.create_raster_dict(pred_gw_file_dir, pattern=pred_gw_pattern)
        years = sorted(list(pred_gw_raster_dict.keys()))
        mean_actual_gw = {}
        mean_pred_gw = {}
        for year in years:
            pred_raster = pred_gw_raster_dict[year]
            pred_raster = pred_raster.reshape(pred_raster.shape[0] * pred_raster.shape[1])
            mean_pred_gw[year] = np.nanmean(pred_raster)
            if year in actual_gw_raster_dict.keys():
                actual_raster = actual_gw_raster_dict[year]
                actual_raster = actual_raster.reshape(actual_raster.shape[0] * actual_raster.shape[1])
                mean_actual_gw[year] = np.nanmean(actual_raster)
                raster_dict = {'YEAR': [year] * actual_raster.shape[0], 'Actual_GW': actual_raster,
                               'Pred_GW': pred_raster}
                if use_gmds:
                    raster_dict['GMD'] = [gmd_name_list[index]] * actual_raster.shape[0]
                gw_raster_df = gw_raster_df.append(pd.DataFrame(data=raster_dict))
            else:
                mean_actual_gw[year] = np.nan
        gw_dict = {'YEAR': years, 'Actual_GW': list(mean_actual_gw.values()), 'Pred_GW': list(mean_pred_gw.values())}
        if use_gmds:
            gw_dict['GMD'] = [gmd_name_list[index]] * len(years)
        gw_df = gw_df.append(pd.DataFrame(data=gw_dict))
    gw_df.to_csv(out_dir + 'gw_yearly_new.csv', index=False)
    gw_raster_df = gw_raster_df.dropna(axis=0)
    gw_raster_df.to_csv(out_dir + 'GW_Raster.csv', index=False)
    return gw_df, gw_raster_df, grace_df


def get_trend(dt_list, value_list):
    """
    Obtain trend data
    :param dt_list:  List of time values as DateTime object
    :param value_list: List of values
    :return: Fitted values
    """

    dt_ordinal = [dt.toordinal() for dt in dt_list]
    trend_model = LinearRegression()
    dt_ordinal = np.array(dt_ordinal).reshape(-1, 1)
    values = np.array(value_list).reshape(-1, 1)
    trend_model.fit(X=dt_ordinal, y=values)
    fit = np.poly1d([trend_model.coef_[0][0]])(value_list)
    return fit


def create_time_series_plot(input_df_list):
    """
    Create time series plot
    :param input_df_list: Input data frames as constructed from #create_gw_time_series
    :return: None
    """

    df1, df2 = input_df_list
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    year = df1['YEAR']
    grace_df = df1[['YEAR', 'GRACE']]
    df1 = df1.drop('GRACE', axis=1)
    df1.set_index('YEAR').plot(ax=ax1)
    df2.set_index('Time').plot(ax=ax3)

    ax1.axvline(x=2011, color='k')
    ax1.set_ylabel('Mean GW Pumping (mm)')
    ax1.set_xticks(year)
    ax1.set_xticklabels(year)
    grace_df.set_index('YEAR').plot(ax=ax2)
    ax2.set_xticks(year)
    ax2.set_xticklabels(year)
    ax2.invert_yaxis()
    ax2.set_ylabel('Mean TWS (mm)')
    ax3.invert_yaxis()
    ax3.set_ylabel('Monthly TWS (mm)')
    plt.show()


def create_time_series_forecast_plot(input_df_list, forecast_years=(2019, ), plot_title='', gmd_train=False):
    """
    Create time series plot
    :param input_df_list: Input data frames as constructed from #create_gw_time_series
    :param forecast_years: The line color changes for these years
    :param plot_title: Plot title
    :param gmd_train: Remove year highlights if gmd_train is set True
    :return: None
    """

    df1, df2 = input_df_list
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(plot_title)
    df1.set_index('YEAR').plot(ax=ax1)
    df2.set_index('DT').plot(ax=ax2)
    df2_years = list(df2.DT)
    ax1.set_xlim(left=np.min(df1.YEAR) - 0.1, right=np.max(df1.YEAR) + 0.1)
    if gmd_train:
        ax1.set_ylim(bottom=0, top=50)
    labels = ['Actual GW', 'Predicted GW']
    bbox_to_anchor = (0.1, 0.3)
    ncol = 3
    if not gmd_train:
        ax1.axvspan(2010.5, 2018.5, color='#a6bddb', alpha=0.6)
        labels += ['Test Years']
        bbox_to_anchor = (0.1, 1)
        ncol = 2
    min_forecast_yr = min(forecast_years)
    labels += ['Forecast']
    ax1.axvspan(min_forecast_yr - 0.5, np.max(df1.YEAR) + 0.1, color='#fee8c8', alpha=1)
    ax1.legend(loc=2, ncol=ncol, frameon=False, fancybox=False, bbox_to_anchor=bbox_to_anchor, labels=labels)
    ax1.set_ylabel('Mean GW Pumping (mm)')
    ax1.set_xticks(df1.YEAR)
    ax1.set_xticklabels(df1.YEAR)
    ax1.set_xlabel('Year')
    ax2.set_ylim(bottom=-150, top=150)
    ax2.invert_yaxis()
    ax2.set_ylabel('Monthly TWS (mm)')
    ax2.set_xlabel('Year')
    ax2.legend(loc=2, bbox_to_anchor=(0.1, 1), frameon=False, fancybox=False, labels=['GRACE TWS'])
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    datemin = np.datetime64(df2_years[0], 'Y')
    datemax = np.datetime64(df2_years[-1], 'Y') + np.timedelta64(1, 'Y')
    ax2.set_xlim(datemin, datemax)
    ax2.format_xdata = mdates.DateFormatter('%Y')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


def create_gmd_time_series_forecast_plot(input_df_list, gmd_name_list, forecast_years=(2019, ), plot_title='',
                                         gmd_train=False):
    """
    Create time series plot considering all GMDs
    :param input_df_list: Input data frames as constructed from #create_gw_time_series
    :param gmd_name_list: GMD labels
    :param forecast_years: The line color changes for these years
    :param plot_title: Plot title
    :param gmd_train: Remove year highlights if gmd_train is set True
    :return: None
    """

    gw_df, grace_df = input_df_list
    for gmd in gmd_name_list:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle(plot_title)
        df = gw_df[gw_df.GMD == gmd]
        df.set_index('YEAR').plot(ax=ax1)
        ax1.set_xlim(left=np.min(df.YEAR) - 0.1, right=np.max(df.YEAR) + 0.1)
        labels = ['Actual GW: ' + gmd, 'Predicted GW: ' + gmd]
        ncol = 3
        bbox_to_anchor = (0.1, 1)
        if gmd == 'GMD3':
            bbox_to_anchor = (0.1, 0.4)
        if not gmd_train:
            ax1.axvspan(2010.5, 2018.5, color='#a6bddb', alpha=0.6)
            labels += ['Test Years']
            ncol = 2
            bbox_to_anchor = (0.1, 1)
        min_forecast_yr = min(forecast_years)
        ax1.axvspan(min_forecast_yr - 0.5, np.max(df.YEAR) + 0.1, color='#fee8c8', alpha=1)
        labels += ['Forecast']
        ax1.legend(loc=2, ncol=ncol, frameon=False, fancybox=False, bbox_to_anchor=bbox_to_anchor, labels=labels)
        ax1.set_ylabel('Mean GW Pumping (mm)')
        ax1.set_xticks(df.YEAR)
        ax1.set_xticklabels(df.YEAR)
        ax1.set_xlabel('Year')
        max_gw = int(max(np.max(gw_df.Actual_GW), np.max(gw_df.Pred_GW)) + 10)
        ax1.set_ylim(bottom=0, top=max_gw)
        grace_df.set_index('DT').plot(ax=ax2)
        ax2.set_ylim(bottom=-150, top=150)
        ax2.invert_yaxis()
        ax2.set_ylabel('Monthly TWS (mm)')
        ax2.set_xlabel('Year')
        ax2.legend(loc=2, bbox_to_anchor=(0.1, 1), frameon=False, fancybox=False, labels=['GRACE TWS'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()


def create_time_series_forecast_plot_multi_pred(input_df_list, forecast_years=(2019, ), plot_title=''):
    """
    Create time series plot considering entire Kansas for predicted GW from All, Only Spatial, and Only Temporal use
    cases
    :param input_df_list: Input data frames as constructed from #run_analysis2
    :param forecast_years: The line color changes for these years
    :param plot_title: Plot title
    :return: None
    """

    df_all, df_spatial, df_st = input_df_list
    gw_df_all = df_all[0]
    gw_df_spatial = df_spatial[0]
    gw_df_st = df_st[0]
    grace_df = df_all[1]
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(plot_title)
    gw_df_all.set_index('YEAR').plot(y='Actual_GW', ax=ax1, color='r', linestyle='solid')
    gw_df_all.set_index('YEAR').plot(y='Pred_GW', ax=ax1, color='#3D357A', linestyle='dashed')
    gw_df_spatial.set_index('YEAR').plot(y='Pred_GW', ax=ax1, color='k', linestyle='solid', linewidth=2)
    gw_df_st.set_index('YEAR').plot(y='Pred_GW', ax=ax1, color='#00A1D9', linestyle=(0, (1, 1)))
    grace_df.set_index('DT').plot(ax=ax2)
    grace_df_years = list(grace_df.DT)
    ax1.axvspan(2010.5, 2018.5, color='#a6bddb', alpha=0.6)
    min_forecast_yr = min(forecast_years)
    ax1.set_xlim(left=np.min(gw_df_all.YEAR) - 0.1, right=np.max(gw_df_all.YEAR) + 0.1)
    ax1.axvspan(min_forecast_yr - 0.5, np.max(gw_df_all.YEAR) + 0.1, color='#fee8c8', alpha=1)
    ax1.legend(loc=2, ncol=2, frameon=False, fancybox=False, bbox_to_anchor=(0.1, 1),
               labels=['Actual GW', 'Predicted GW: All', 'Predicted GW: Spatial', 'Predicted GW: ST',
                       'Test Years', 'Forecast'])
    ax1.set_ylabel('Mean GW Pumping (mm)')
    ax1.set_xticks(gw_df_all.YEAR)
    ax1.set_xticklabels(gw_df_all.YEAR)
    ax1.set_xlabel('Year')
    ax2.set_ylim(bottom=-150, top=150)
    ax2.invert_yaxis()
    ax2.set_ylabel('Monthly TWS (mm)')
    ax2.set_xlabel('Year')
    ax2.legend(loc=2, bbox_to_anchor=(0.1, 1), frameon=False, fancybox=False, labels=['GRACE TWS'])
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    datemin = np.datetime64(grace_df_years[0], 'Y')
    datemax = np.datetime64(grace_df_years[-1], 'Y') + np.timedelta64(1, 'Y')
    ax2.set_xlim(datemin, datemax)
    ax2.format_xdata = mdates.DateFormatter('%Y')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()


def create_gmd_time_series_forecast_plot_multi_pred(input_df_list, gmd_name_list, forecast_years=(2019, ),
                                                    plot_title=''):
    """
    Create time series plot considering all GMDs for predicted GW from All, Only Spatial, and Only Temporal use cases
    :param input_df_list: Input data frames as constructed from #run_analysis2
    :param gmd_name_list: GMD labels
    :param forecast_years: The line color changes for these years
    :param plot_title: Plot title
    :return: None
    """

    df_all, df_spatial, df_st = input_df_list
    gw_df_all = df_all[0]
    gw_df_spatial = df_spatial[0]
    gw_df_st = df_st[0]
    grace_df = df_all[1]
    for gmd in gmd_name_list:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle(plot_title)
        df_all = gw_df_all[gw_df_all.GMD == gmd]
        df_spatial = gw_df_spatial[gw_df_spatial.GMD == gmd]
        df_st = gw_df_st[gw_df_st.GMD == gmd]
        df_all.set_index('YEAR').plot(y='Actual_GW', ax=ax1, color='r', linestyle='solid')
        df_all.set_index('YEAR').plot(y='Pred_GW', ax=ax1, color='#3D357A', linestyle='dashed')
        df_spatial.set_index('YEAR').plot(y='Pred_GW', ax=ax1, color='k', linestyle='solid', linewidth=2)
        df_st.set_index('YEAR').plot(y='Pred_GW', ax=ax1, color='#00A1D9', linestyle=(0, (1, 1)))
        ax1.axvspan(2010.5, 2018.5, color='#a6bddb', alpha=0.6)
        min_forecast_yr = min(forecast_years)
        ax1.set_xlim(left=np.min(df_all.YEAR) - 0.1, right=np.max(df_all.YEAR) + 0.1)
        ax1.axvspan(min_forecast_yr - 0.5, np.max(df_all.YEAR) + 0.1, color='#fee8c8', alpha=1)
        bbox_to_anchor = (0.01, 1)
        if gmd == 'GMD3':
            bbox_to_anchor = (0.01, 0.4)
        ax1.legend(loc=2, ncol=2, frameon=False, fancybox=False, bbox_to_anchor=bbox_to_anchor,
                   labels=['Actual GW: ' + gmd, 'Predicted GW (All): ' + gmd, 'Predicted GW (Spatial): ' + gmd,
                           'Predicted GW (ST): ' + gmd, 'Test Years', 'Forecast'])
        ax1.set_ylabel('Mean GW Pumping (mm)')
        ax1.set_xticks(df_all.YEAR)
        ax1.set_xticklabels(df_all.YEAR)
        ax1.set_xlabel('Year')
        max_gw = int(max(np.max(gw_df_all.Actual_GW), np.max(gw_df_all.Pred_GW)) + 10)
        ax1.set_ylim(bottom=0, top=max_gw)
        grace_df.set_index('DT').plot(ax=ax2)
        ax2.set_ylim(bottom=-150, top=150)
        ax2.invert_yaxis()
        ax2.set_ylabel('Monthly TWS (mm)')
        ax2.set_xlabel('Year')
        ax2.legend(loc=2, bbox_to_anchor=(0.01, 1), frameon=False, fancybox=False, labels=['GRACE TWS'])
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()


def preprocess_gmds(actual_gw_dir, pred_gw_dir, input_gmd_file, out_dir, actual_gw_pattern='GW*.tif',
                    pred_gw_pattern='pred*.tif'):
    """
    Preprocess GMD shapefiles and rasters
    :param actual_gw_dir: Directory containing the actual data
    :param pred_gw_dir: Directory containing the predicted data
    :param input_gmd_file: Input GMD shapefile for creating GMD specific plots
    :param out_dir: Output directory for storing intermediate files
    :param actual_gw_pattern: Actual GW pumping raster file pattern
    :param pred_gw_pattern: Predicted GW pumping raster file pattern
    :return: Directory paths (as tuple) of the actual and predicted GMD specific GW rasters along with the GMD names
    """

    out_shp_dir = make_proper_dir_name(out_dir + 'GMD_SHP')
    out_actual_gmd_dir = make_proper_dir_name(out_dir + 'Actual_GMD_Rasters')
    out_pred_gmd_dir = make_proper_dir_name(out_dir + 'Pred_GMD_Rasters')
    makedirs([out_shp_dir, out_actual_gmd_dir, out_pred_gmd_dir])
    vops.extract_polygons(input_gmd_file, out_shp_dir)
    actual_gw_dir_list, pred_gw_dir_list, gmd_name_list = [], [], []
    print('Preprocessing started...')
    for gmd_shp in glob(out_shp_dir + '*.shp'):
        gmd_name = gmd_shp[gmd_shp.rfind(os.sep) + 1: gmd_shp.rfind('.')]
        gmd_name_list.append(gmd_name)
        actual_outdir = make_proper_dir_name(out_actual_gmd_dir + gmd_name)
        pred_outdir = make_proper_dir_name(out_pred_gmd_dir + gmd_name)
        makedirs([actual_outdir, pred_outdir])
        actual_gw_dir_list.append(actual_outdir)
        pred_gw_dir_list.append(pred_outdir)
        rops.crop_rasters(input_raster_dir=actual_gw_dir, input_mask_file=gmd_shp, pattern=actual_gw_pattern,
                          outdir=actual_outdir, ext_mask=False)
        rops.crop_rasters(input_raster_dir=pred_gw_dir, input_mask_file=gmd_shp, pattern=pred_gw_pattern,
                          outdir=pred_outdir, ext_mask=False)
    return actual_gw_dir_list, pred_gw_dir_list, gmd_name_list


def preprocess_gmd_train_test(actual_gw_dir_list, pred_gw_dir_list, input_train_shp, out_dir, forecast_years=(2019,),
                              rev_train_test=False, gmd_exclude_list=None):
    """
    Preprocess GMD rasters to obtain train and test rasters for each GMD based on input_train_shp.
    #preprocess_gmds() must be called before using this method.
    :param actual_gw_dir_list: Actual GW directory list obtained from #preprocess_gmds()
    :param pred_gw_dir_list: Predicted GW directory list obtained from #preprocess_gmds()
    :param input_train_shp: Training shape file path
    :param out_dir: Output directory for storing intermediate files
    :param forecast_years: Tuple of forecast years
    :param rev_train_test: Set True to swap train and test arrays (required when input_train_shp_file is actually used
    for testing instead of training)
    :param gmd_exclude_list: Exclude these GMDs from preprocessing
    :return: Directory paths (as tuple) of the actual and predicted GMD specific train and test GW rasters
    """

    gw_dir_lists = [actual_gw_dir_list, pred_gw_dir_list]
    gw_labels = ['Actual_GW', 'Pred_GW']
    gmd_df_list = [pd.DataFrame(), pd.DataFrame()]
    gmd_df_forecast = pd.DataFrame()
    for pos, gw_dir_list in enumerate(gw_dir_lists):
        gw_dir_list.sort()
        for gw_dir in gw_dir_list:
            gmd_name = gw_dir[gw_dir[:-1].rfind(os.sep) + 1: gw_dir.rfind(os.sep)]
            if gmd_name not in gmd_exclude_list:
                raster_file_list = glob(gw_dir + '*.tif')
                raster_file_list.sort()
                for raster_file in raster_file_list:
                    year = raster_file[raster_file.rfind('_') + 1: raster_file.rfind('.')]
                    train_arr, test_arr = rops.extract_train_test_raster_arr(raster_file, input_train_shp,
                                                                             rev_train_test)
                    train_arr = train_arr.reshape(train_arr.shape[0] * train_arr.shape[1]).tolist()
                    test_arr = test_arr.reshape(test_arr.shape[0] * test_arr.shape[1]).tolist()
                    train_label, test_label = ['TRAIN'] * len(train_arr), ['TEST'] * len(test_arr)
                    raster_arr = train_arr + test_arr
                    raster_label = train_label + test_label
                    year_list = [year] * len(raster_arr)
                    gmd_list = [gmd_name] * len(raster_arr)
                    raster_dict = {'YEAR': year_list, 'DATA': raster_label, gw_labels[pos]: raster_arr, 'GMD': gmd_list}
                    df = pd.DataFrame(data=raster_dict)
                    if int(year) not in forecast_years:
                        gmd_df_list[pos] = gmd_df_list[pos].append(df)
                    else:
                        df['DATA'] = ['FORECAST'] * len(raster_label)
                        gmd_df_forecast = gmd_df_forecast.append(df)
                        gmd_df_forecast = gmd_df_forecast.dropna()
    gmd_df_actual = gmd_df_list[0]
    gmd_df_pred = gmd_df_list[1]
    gmd_df_pred[gw_labels[0]] = gmd_df_actual[gw_labels[0]]
    gmd_df_pred = gmd_df_pred.dropna()
    gmd_df_pred = gmd_df_pred.append(gmd_df_forecast)
    out_csv = out_dir + 'GMD_Train_Test.csv'
    with open(out_csv, 'a+') as out_file:
        gmd_df_pred.to_csv(out_file, mode='a+', header=out_file.tell() == 0, index=False)
    pd.read_csv(out_csv).to_csv(out_csv, index=False)
    return gmd_df_pred


def calculate_gmd_stats_yearly(gw_df, gmd_name_list, out_dir, train_end=2010, test_start=2011):
    """
    Calculate error metrics for each GMD
    :param gw_df: Input GW Dataframe containing actual and predicted GW
    :param gmd_name_list: GMD labels
    :param out_dir: Output directory to write results to
    :param train_end: Training year end
    :param test_start: Test year start
    :return: GMD metrics dataframe
    """

    gmd_metrics_df = pd.DataFrame()
    gw_df = gw_df.dropna(axis=0)
    gw_df_list = gw_df[gw_df.YEAR <= train_end], gw_df[gw_df.YEAR >= test_start], gw_df
    gw_df_labels = ['TRAIN', 'TEST', 'ALL']
    for gw_df, gw_df_label in zip(gw_df_list, gw_df_labels):
        for gmd in gmd_name_list:
            actual_values = gw_df[gw_df.GMD == gmd].Actual_GW
            pred_values = gw_df[gw_df.GMD == gmd].Pred_GW
            rmse = metrics.mean_squared_error(actual_values, pred_values, squared=False)
            r2 = np.round(metrics.r2_score(actual_values, pred_values), 2)
            mae = metrics.mean_absolute_error(actual_values, pred_values)
            nmae = np.round(mae / np.mean(actual_values), 2)
            nrmse = np.round(rmse / np.mean(actual_values), 2)
            mae = np.round(mae, 2)
            rmse = np.round(rmse, 2)
            gmd_metrics_dict = {'GW_TYPE': [gw_df_label], 'GMD': [gmd], 'RMSE': [rmse], 'R2': [r2], 'MAE': [mae],
                                'NMAE': [nmae], 'NRMSE': [nrmse]}
            gmd_metrics_df = gmd_metrics_df.append(pd.DataFrame(data=gmd_metrics_dict))
    out_csv = out_dir + 'GMD_Metrics.csv'
    gmd_metrics_df.to_csv(out_csv, index=False)
    return gmd_metrics_df


def get_error_stats(actual_values, pred_values, round_places=2):
    """
    Get R2 (scikit-learn), Standard R2 (coefficient of determination), MAE, RMSE, NMAE (normalized MAE), and
    NRMSE (normalized RMSE)
    :param actual_values: List of actual values
    :param pred_values: List of predicted values
    :param round_places: Number of decimal places to round at, default 2.
    :return: Tuple containing R2, Standard R2, MAE, RMSE, NMAE, and NRMSE (rounded to 2 decimal places by default)
    """

    if isinstance(actual_values, pd.DataFrame):
        actual_values = actual_values.iloc[:, 0].tolist()
    if isinstance(pred_values, pd.DataFrame):
        pred_values = pred_values.iloc[:, 0].tolist()
    mae = metrics.mean_absolute_error(actual_values, pred_values)
    r2_score = np.round(metrics.r2_score(actual_values, pred_values), round_places)
    standard_r2 = np.round(np.corrcoef(actual_values, pred_values)[0, 1] ** 2, round_places)
    rmse = metrics.mean_squared_error(actual_values, pred_values, squared=False)
    nrmse = np.round(rmse / np.mean(actual_values), round_places)
    nmae = np.round(mae / np.mean(actual_values), round_places)
    rmse = np.round(rmse, round_places)
    mae = np.round(mae, round_places)
    return r2_score, standard_r2, mae, rmse, nmae, nrmse


def calculate_gmd_stats_train_test(gw_df, out_dir):
    """
    Calculate error metrics for both train and test GMDs
    :param gw_df: Input GW Dataframe containing actual and predicted GW
    :param out_dir: Output directory to write results to
    :return: Tuple of GMD, Yearly GMD, and Mean GMD metrics dataframes
    """

    gmd_name_list = set(gw_df['GMD'])
    gmd_metrics, gmd_metrics_yearly, mean_gmd_metrics = [pd.DataFrame()] * 3
    for gmd in gmd_name_list:
        gmd_df = gw_df[gw_df.GMD == gmd]
        train_df = gmd_df[gmd_df.DATA == 'TRAIN']
        test_df = gmd_df[gmd_df.DATA == 'TEST']
        all_df = gmd_df[gmd_df.DATA != 'FORECAST']
        df_list1 = [train_df, test_df, all_df]
        df_names = ['TRAIN', 'TEST', 'TRAIN+TEST']
        for df, df_name in zip(df_list1, df_names):
            actual_gw = df['Actual_GW']
            pred_gw = df['Pred_GW']
            r2_score, standard_r2, mae, rmse, nmae, nrmse = get_error_stats(actual_gw, pred_gw)
            gmd_metrics = gmd_metrics.append(pd.DataFrame(data={'GMD': [gmd], 'DATA': [df_name], 'R2': [r2_score],
                                                                'SR2': [standard_r2], 'MAE': [mae], 'RMSE': [rmse],
                                                                'NMAE': [nmae], 'NRMSE': [nrmse]}))
        year_list = set(train_df['YEAR'])
        for year in year_list:
            train_df_yearly = train_df[train_df.YEAR == year]
            test_df_yearly = test_df[test_df.YEAR == year]
            all_df_yearly = all_df[all_df.YEAR == year]
            df_list2 = [train_df_yearly, test_df_yearly, all_df_yearly]
            for df, df_name in zip(df_list2, df_names):
                actual_gw = df['Actual_GW']
                pred_gw = df['Pred_GW']
                mean_actual_gw = np.mean(actual_gw)
                mean_pred_gw = np.mean(pred_gw)
                r2_score, standard_r2, mae, rmse, nmae, nrmse = get_error_stats(actual_gw, pred_gw)
                gmd_metrics_yearly = gmd_metrics_yearly.append(pd.DataFrame(data={'YEAR': [year], 'GMD': [gmd],
                                                                                  'DATA': [df_name],
                                                                                  'Actual_GW': [mean_actual_gw],
                                                                                  'Pred_GW': [mean_pred_gw],
                                                                                  'R2': [r2_score],
                                                                                  'SR2': [standard_r2], 'MAE': [mae],
                                                                                  'RMSE': [rmse], 'NMAE': [nmae],
                                                                                  'NRMSE': [nrmse]}))

        for df, df_name in zip(df_list1, df_names):
            gmd_metrics_df = gmd_metrics_yearly[(gmd_metrics_yearly.DATA == df_name) & (gmd_metrics_yearly.GMD == gmd)]
            actual_gw = gmd_metrics_df['Actual_GW']
            pred_gw = gmd_metrics_df['Pred_GW']
            r2_score, standard_r2, mae, rmse, nmae, nrmse = get_error_stats(actual_gw, pred_gw)
            mean_gmd_metrics = mean_gmd_metrics.append(pd.DataFrame(data={'GMD': [gmd], 'DATA': [df_name],
                                                                          'R2': [r2_score], 'SR2': [standard_r2],
                                                                          'MAE': [mae], 'RMSE': [rmse], 'NMAE': [nmae],
                                                                          'NRMSE': [nrmse]}))
    forecast_data = gw_df[gw_df.DATA == 'FORECAST']
    forecast_years = set(forecast_data['YEAR'])
    forecast_df = pd.DataFrame()
    for gmd in gmd_name_list:
        f_df = forecast_data[forecast_data.GMD == gmd]
        for year in forecast_years:
            fy_df = f_df[forecast_data.YEAR == year]
            mean_forecast_gw = np.mean(fy_df['Pred_GW'])
            forecast_df = forecast_df.append(pd.DataFrame(data={'YEAR': [year], 'GMD': [gmd], 'DATA': ['TEST'],
                                                                'Actual_GW': [np.nan], 'Pred_GW': [mean_forecast_gw],
                                                                'R2': [np.nan], 'SR2': [np.nan], 'MAE': [np.nan],
                                                                'RMSE': [np.nan], 'NMAE': [np.nan],
                                                                'NRMSE': [np.nan]}))
    gmd_metrics_yearly = gmd_metrics_yearly.append(forecast_df)
    gmd_metrics = gmd_metrics.sort_values(by=['DATA', 'GMD'])
    gmd_metrics_yearly = gmd_metrics_yearly.sort_values(by=['YEAR', 'DATA', 'GMD'])
    mean_gmd_metrics = mean_gmd_metrics.sort_values(by=['DATA', 'GMD'])
    gmd_metrics_out = out_dir + 'GMD_Metrics_Train_Test.csv'
    gmd_metrics_yearly_out = out_dir + 'GMD_Metrics_Train_Test_Yearly.csv'
    mean_gmd_metrics_out = out_dir + 'Mean_GMD_Metrics_Train_Test.csv'
    out_files = [gmd_metrics_out, gmd_metrics_yearly_out, mean_gmd_metrics_out]
    out_df = [gmd_metrics, gmd_metrics_yearly, mean_gmd_metrics]
    for out_file, out_df in zip(out_files, out_df):
        with open(out_file, 'a+') as of:
            out_df.to_csv(of, mode='a+', header=of.tell() == 0, index=False)
        pd.read_csv(out_file).to_csv(out_file, index=False)
    return gmd_metrics, gmd_metrics_yearly, mean_gmd_metrics


def run_analysis(actual_gw_dir, pred_gw_dir, grace_csv, out_dir, input_gmd_file=None, use_gmds=True,
                 actual_gw_pattern='GW*.tif', pred_gw_pattern='pred*.tif', generate_plots=True, gmd_train=False,
                 input_train_shp_file=None, rev_train_test=False, gmd_exclude_list=None, load_gmd_train_test_csv=False):
    """
    Run model analysis to get actual vs predicted graph along with GRACE TWSA variations
    :param actual_gw_dir: Directory containing the actual data
    :param pred_gw_dir: Directory containing the predicted data
    :param grace_csv: GRACE TWSA CSV file
    :param out_dir: Output directory for storing intermediate files
    :param input_gmd_file: Input GMD shapefile for creating GMD specific plots, required only if use_gmds=True
    :param use_gmds: Set False to use entire GW raster for analysis
    :param actual_gw_pattern: Actual GW pumping raster file pattern
    :param pred_gw_pattern: Predicted GW pumping raster file pattern
    :param generate_plots: Set False to disable plotting
    :param gmd_train: Set True to use custom shapefile for getting GMD specific results
    :param input_train_shp_file: Training shape file path, gmd_train must be True
    :param rev_train_test: Set True to swap train and test arrays (required when input_train_shp_file is actually used
    for testing instead of training), works only if gmd_train=True
    :param gmd_exclude_list: Exclude these GMDs from preprocessing
    :param load_gmd_train_test_csv: Set True to directly load GMD Train Test CSV which has been previously created for
    plotting, this should be set to True only when all the GMD data are present in the CSV
    :return: Tuple of Pandas dataframes if generate_plots=False else None
    """

    out_dir = make_proper_dir_name(out_dir)
    gmd_name_list = []
    ts_df = pd.DataFrame()
    makedirs([out_dir])
    if not use_gmds:
        ts_df = create_gw_forecast_time_series([actual_gw_dir], [pred_gw_dir], grace_csv=grace_csv, out_dir=out_dir,
                                               actual_gw_pattern=actual_gw_pattern, pred_gw_pattern=pred_gw_pattern,
                                               use_gmds=use_gmds)
        ts_df = ts_df[0], ts_df[2]
        if generate_plots:
            create_time_series_forecast_plot(ts_df, gmd_train=gmd_train)
    else:
        actual_gw_dir_list, pred_gw_dir_list, gmd_name_list = preprocess_gmds(actual_gw_dir, pred_gw_dir,
                                                                              input_gmd_file, out_dir,
                                                                              actual_gw_pattern, pred_gw_pattern)
        if gmd_train:
            if not load_gmd_train_test_csv:
                gw_df = preprocess_gmd_train_test(actual_gw_dir_list, pred_gw_dir_list, input_train_shp_file, out_dir,
                                                  rev_train_test=rev_train_test, gmd_exclude_list=gmd_exclude_list)
                calculate_gmd_stats_train_test(gw_df, out_dir)
            else:
                grace_df = create_gw_forecast_time_series(None, None, grace_csv, get_only_grace_df=True)
                gw_df = pd.read_csv(out_dir + 'GMD_Metrics_Train_Test_Yearly.csv')
                gw_df = gw_df[gw_df.DATA == 'TEST'][['YEAR', 'GMD', 'Actual_GW', 'Pred_GW']]
                create_gmd_time_series_forecast_plot((gw_df, grace_df), gmd_name_list=gmd_name_list,
                                                     gmd_train=gmd_train)
        else:
            ts_df = create_gw_forecast_time_series(actual_gw_dir_list, pred_gw_dir_list, gmd_name_list=gmd_name_list,
                                                   grace_csv=grace_csv, use_gmds=use_gmds, out_dir=out_dir,
                                                   actual_gw_pattern=actual_gw_pattern, pred_gw_pattern=pred_gw_pattern)
            gw_df = ts_df[1]
            ts_df = ts_df[0], ts_df[2]
            if generate_plots:
                print(calculate_gmd_stats_yearly(gw_df, gmd_name_list, out_dir))
                create_gmd_time_series_forecast_plot(ts_df, gmd_name_list=gmd_name_list)
    if (not gmd_train) and (not generate_plots):
        if use_gmds:
            return ts_df, gmd_name_list
        return ts_df


def generate_feature_qq_plots(input_csv_file, year_col='YEAR', temporal_features=('ET', 'P'), pred_attr='GW'):
    """
    Generate QQ plots for all features
    :param input_csv_file: Input CSV file path
    :param year_col: Name of Year column
    :param temporal_features: Temporal feature names
    :param pred_attr: Prediction attribute name to be dropped from boxplot
    :return: None
    """

    input_df = pd.read_csv(input_csv_file)
    feature_names = input_df.columns.values.tolist()
    feature_names.remove(pred_attr)
    for tf in temporal_features:
        sub_df = input_df[[year_col, tf]]
        fig, ax = plt.subplots(figsize=(12, 5))
        seaborn.boxplot(x='YEAR', y=tf, data=sub_df, ax=ax)
        plt.show()
        feature_names.remove(tf)
    feature_names.remove(year_col)
    feature_names.remove('Crop')
    sub_df = pd.melt(input_df.loc[input_df[year_col] == 2015][feature_names])
    sub_df = sub_df.rename(columns={'variable': 'Land-Use Features', 'value': 'Land-Use Density'})
    seaborn.boxplot(x='Land-Use Features', y='Land-Use Density', data=sub_df)
    plt.show()
    sub_df = pd.melt(input_df.loc[input_df[year_col] == 2015][['Crop']])
    sub_df['variable'] = ''
    sub_df = sub_df.rename(columns={'variable': 'Crop Coefficient', 'value': 'Value'})
    seaborn.boxplot(x='Crop Coefficient', y='Value', data=sub_df)
    plt.show()


def run_analysis2(actual_gw_dir, pred_gw_dir_list, grace_csv, out_dir, input_gmd_file=None, use_gmds=True,
                  actual_gw_pattern='GW*.tif', pred_gw_pattern='pred*.tif'):
    """
    Run model analysis to get actual vs predicted graph along with GRACE TWSA variations for different use cases
    :param actual_gw_dir: Directory containing the actual data
    :param pred_gw_dir_list: List of directories containing the predicted data for different use-cases ordered by All,
    Only Spatial, and Only Spatio-temporal predictions
    :param grace_csv: GRACE TWSA CSV file
    :param out_dir: Output directory for storing intermediate files
    :param input_gmd_file: Input GMD shapefile for creating GMD specific plots, required only if use_gmds=True
    :param use_gmds: Set False to use entire GW raster for analysis
    :param actual_gw_pattern: Actual GW pumping raster file pattern
    :param pred_gw_pattern: Predicted GW pumping raster file pattern
    :return: None
    """

    ts_list = []
    gmd_name_list = []
    for pred_gw_dir in pred_gw_dir_list:
        ts_df = run_analysis(actual_gw_dir, pred_gw_dir, grace_csv, out_dir, input_gmd_file, use_gmds,
                             actual_gw_pattern, pred_gw_pattern, generate_plots=False)
        if use_gmds:
            ts_df, gmd_name_list = ts_df
        ts_list.append(ts_df)
    if not use_gmds:
        create_time_series_forecast_plot_multi_pred(ts_list)
    else:
        create_gmd_time_series_forecast_plot_multi_pred(ts_list, gmd_name_list=gmd_name_list)
