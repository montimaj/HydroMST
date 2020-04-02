from Python_Files import rasterops as rops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_gw_time_series(actual_gw_file_dir, pred_gw_file_dir, grace_dir, actual_gw_pattern='GW*.tif',
                          pred_gw_pattern='pred*.tif', grace_pattern='GRACE*.tif', out_csv='../Output/analysis.csv'):
    """
    Create time series data for actual and predicted GW values (annual mean) along with GRACE
    :param actual_gw_file_dir: Actual GW pumping raster directory
    :param pred_gw_file_dir: Predicted GW pumping raster directory
    :param grace_dir: GRACE raster directory
    :param actual_gw_pattern: Actual GW pumping raster file pattern
    :param pred_gw_pattern: Predicted GW pumping raster file pattern
    :param grace_pattern: GRACE raster file pattern
    :param out_csv: Output csv file to be generated from the dataframe
    :return: Dataframe containing year and corresponding mean values
    """

    actual_gw_raster_dict = rops.create_raster_dict(actual_gw_file_dir, pattern=actual_gw_pattern)
    pred_gw_raster_dict = rops.create_raster_dict(pred_gw_file_dir, pattern=pred_gw_pattern)
    grace_raster_dict = rops.create_raster_dict(grace_dir, pattern=grace_pattern)
    years = sorted(list(pred_gw_raster_dict.keys()))
    mean_actual_gw = {}
    mean_pred_gw = {}
    mean_grace = {}
    for year in years:
        mean_actual_gw[year] = np.nanmean(actual_gw_raster_dict[year])
        mean_pred_gw[year] = np.nanmean(pred_gw_raster_dict[year])
        mean_grace[year] = np.nanmean(grace_raster_dict[year])
    df_dict = {'YEAR': years, 'Actual_GW': list(mean_actual_gw.values()), 'Pred_GW': list(mean_pred_gw.values()),
               'GRACE': list(mean_grace.values())}
    df = pd.DataFrame(data=df_dict)
    df.to_csv(out_csv, index=False)
    return df


def create_time_series_plot(input_df):
    """
    Create time series plot
    :param input_df: Input data frame as constructed from #create_gw_time_series
    :return: None
    """
    fig, (ax1, ax2) = plt.subplots(2, 1)
    year = input_df['YEAR']
    grace_df = input_df[['YEAR', 'GRACE']]
    input_df = input_df.drop('GRACE', axis=1)
    input_df.set_index('YEAR').plot(ax=ax1)
    ax1.axvline(x=2011, color='k')
    ax1.set_ylabel('Mean GW Pumping (mm)')
    ax1.set_xticks(year)
    ax1.set_xticklabels(year)
    grace_df.set_index('YEAR').plot(ax=ax2)
    ax2.set_xticks(year)
    ax2.set_xticklabels(year)
    ax2.invert_yaxis()
    ax2.set_ylabel('Mean TWS Change (mm)')
    plt.show()


gw_dir = '../Files/RF_Data_All/'
pred_gw_dir = '../Output/Predicted_Rasters_All/'
grace_dir = '../Files/Masked_Rasters_All/GRACE_DA_Scaled/'
ts_df = create_gw_time_series(gw_dir, pred_gw_dir, grace_dir)
create_time_series_plot(ts_df)
