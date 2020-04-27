# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Python_Files.hydrolibs import rasterops as rops
from datetime import datetime
from sklearn.linear_model import LinearRegression


def create_gw_time_series(actual_gw_file_dir, pred_gw_file_dir, grace_dir, actual_gw_pattern='GW*.tif',
                          pred_gw_pattern='pred*.tif', grace_pattern='GRACE*.tif', make_trend=False,
                          out_dir='../Output/'):
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


def create_gw_forecast_time_series(actual_gw_file_dir, pred_gw_file_dir, grace_csv, actual_gw_pattern='GW*.tif',
                                   pred_gw_pattern='pred*.tif', out_dir='../Output/', forecast_years=(2019, )):
    """
    Create GW and GRACE dataframes
    :param actual_gw_file_dir: Actual GW pumping raster directory
    :param pred_gw_file_dir: Predicted GW pumping raster directory
    :param grace_csv: GRACE TWS CSV file
    :param actual_gw_pattern: Actual GW pumping raster file pattern
    :param pred_gw_pattern: Predicted GW pumping raster file pattern
    :param out_dir: Output directory for storing the CSV files
    :param forecast_years: Years used as forecast data (these don't have actual observations)
    :return: Two dataframes, one with the GW pumping values and the other containing the monthly GRACE values
    """

    actual_gw_raster_dict = rops.create_raster_dict(actual_gw_file_dir, pattern=actual_gw_pattern)
    pred_gw_raster_dict = rops.create_raster_dict(pred_gw_file_dir, pattern=pred_gw_pattern)
    grace_df = pd.read_csv(grace_csv)
    grace_df = grace_df.dropna(axis=0)
    grace_df['GRACE'] = grace_df['GRACE'] * 10
    grace_df['DT'] = pd.to_datetime(grace_df['DT']).dt.date
    years = sorted(list(pred_gw_raster_dict.keys()))
    mean_actual_gw = {}
    mean_pred_gw = {}
    for year in years:
        mean_pred_gw[year] = np.nanmean(pred_gw_raster_dict[year])
        if year in actual_gw_raster_dict.keys():
            mean_actual_gw[year] = np.nanmean(actual_gw_raster_dict[year])
        else:
            mean_actual_gw[year] = mean_pred_gw[year]
    gw_df = {'YEAR': years, 'Actual_GW': list(mean_actual_gw.values()), 'Pred_GW': list(mean_pred_gw.values())}
    gw_df = pd.DataFrame(data=gw_df)
    gw_df.to_csv(out_dir + 'gw_yearly_new.csv', index=False)
    return gw_df, grace_df


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


def create_time_series_forecast_plot(input_df_list, forecast_years=(2019, )):
    """
    Create time series plot
    :param input_df_list: Input data frames as constructed from #create_gw_time_series
    :param forecast_years: The line color changes for these years
    :return: None
    """

    df1, df2 = input_df_list
    fig, (ax1, ax2) = plt.subplots(2, 1)
    year = df1['YEAR']
    df1.set_index('YEAR').plot(ax=ax1)
    df2.set_index('DT').plot(ax=ax2)

    ax1.axvline(x=2011, color='k', linestyle='--')
    min_forecast_yr = min(forecast_years)
    ax1.axvline(x=min_forecast_yr - 1, color='r', linestyle='--')
    ax1.legend(loc=2, ncol=2, frameon=False, fancybox=False, bbox_to_anchor=(0.1, 1),
               labels=['Actual GW', 'Pred GW', 'Test Data (2011-2018)', 'Forecast'])
    ax1.set_ylabel('Mean GW Pumping (mm)')
    ax1.set_xticks(year)
    ax1.set_xticklabels(year)
    ax1.set_xlabel('')
    ax2.invert_yaxis()
    ax2.set_ylabel('Monthly TWS (mm)')
    ax2.set_xlabel('Year')
    ax2.legend(loc=2, bbox_to_anchor=(0.1, 1), frameon=False, fancybox=False, labels=['GRACE TWS'])
    plt.show()


gw_dir = '../Files_New/RF_Data_All/'
pred_gw_dir = '../Output/Predicted_Rasters_All/'
grace_csv = '../Data/GRACE/TWS_GRACE.csv'
# grace_dir = '../Files/Masked_Rasters_All/GRACE_Scaled/'
# ts_df = create_gw_time_series(gw_dir, pred_gw_dir, grace_dir)
ts_df = create_gw_forecast_time_series(gw_dir, pred_gw_dir, grace_csv)
create_time_series_forecast_plot(ts_df)
