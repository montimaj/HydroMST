from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from Python_Files import rasterops as rops


def create_dataframe(input_file_dir, out_df, pattern='*.tif', exclude_years=(), exclude_vars=(), make_year_col=True):
    """
    Create dataframe from file list
    :param input_file_dir: Input directory where the file names begin with <Variable>_<Year>, e.g, ET_2015.tif
    :param out_df: Output Dataframe file
    :param pattern: File pattern to look for in the folder
    :param exclude_years: Exclude these years from the dataframe
    :param exclude_vars: Exclude these variables from the dataframe
    :param make_year_col: Make a dataframe column entry for year
    :return: Pandas dataframe
    """

    raster_file_dict = defaultdict(lambda: [])
    for f in glob(input_file_dir + pattern):
        sep = f.rfind('_')
        variable, year = f[f.rfind('/') + 1: sep], f[sep + 1: f.rfind('.')]
        if variable not in exclude_vars and int(year) not in exclude_years:
            raster_file_dict[int(year)].append(f)

    raster_dict = {}
    flag = False
    years = [yr for yr in raster_file_dict.keys()]
    years.sort()
    for year in years:
        file_list = raster_file_dict[year]
        for raster_file in file_list:
            raster_arr = rops.read_raster_as_arr(raster_file, get_file=False)
            raster_arr = raster_arr.reshape(raster_arr.shape[0] * raster_arr.shape[1])
            variable = raster_file[raster_file.rfind('/') + 1: raster_file.rfind('_')]
            raster_dict[variable] = raster_arr
        if make_year_col:
            raster_dict['YEAR'] = [year] * raster_arr.shape[0]
        if not flag:
            df = pd.DataFrame(data=raster_dict)
            flag = True
        else:
            df = df.append(pd.DataFrame(data=raster_dict))

    df = df.dropna(axis=0)
    df.to_csv(out_df, index=False)
    return df


def split_data_train_test(input_df, pred_attr='GW_KS', shuffle=True, random_state=0, test_size=0.2, outdir=None,
                          exclude_year=False):
    """
    Split data preserving temporal variations
    :param input_df: Input dataframe
    :param pred_attr: Prediction attribute name
    :param shuffle: Default True for shuffling
    :param random_state: Random state used during train test split
    :param test_size: Test data size percentage (0<=test_size<=1)
    :param outdir: Set path to store intermediate files
    :param exclude_year: Set true for not adding the year attribute for model building and prediction
    :return: X_train, X_test, y_train, y_test
    """

    years = set(input_df['YEAR'])
    x_train_df = pd.DataFrame()
    x_test_df = pd.DataFrame()
    y_train_df = pd.DataFrame()
    y_test_df = pd.DataFrame()
    for year in years:
        selected_data = input_df.loc[input_df['YEAR'] == year]
        y = selected_data[pred_attr]
        drop_columns = [pred_attr]
        if exclude_year:
            drop_columns += ['YEAR']
        selected_data = selected_data.drop(columns=drop_columns)
        x_train, x_test, y_train, y_test = train_test_split(selected_data, y, shuffle=shuffle,
                                                            random_state=random_state, test_size=test_size)
        x_train_df = x_train_df.append(x_train)
        x_test_df = x_test_df.append(x_test)
        y_train_df = pd.concat([y_train_df, y_train])
        y_test_df = pd.concat([y_test_df, y_test])

    if outdir:
        x_train_df.to_csv(outdir + 'X_Train.csv', index=False)
        x_test_df.to_csv(outdir + 'X_Test.csv', index=False)
        y_train_df.to_csv(outdir + 'Y_Train.csv', index=False)
        y_test_df.to_csv(outdir + 'Y_Test.csv', index=False)

    return x_train_df, x_test_df, y_train_df[0].ravel(), y_test_df[0].ravel()


def rf_regressor(input_df, out_dir, n_estimators=200, random_state=0, test_size=0.2, pred_attr='GW_KS', shuffle=True,
                 plot_graphs=False, exclude_year=False):
    """
    Perform random forest regression
    :param input_df: Input pandas dataframe
    :param out_dir: Output file directory for storing intermediate results
    :param n_estimators: RF hyperparameter
    :param random_state: RF hyperparameter
    :param test_size: RF hyperparameter
    :param pred_attr: Prediction attribute name in the dataframe
    :param shuffle: Set False to stop data shuffling
    :param plot_graphs: Plot Actual vs Prediction graph
    :param exclude_year: Set true for not adding the year attribute for model building and prediction
    :return: Random forest model
    """

    x_train, x_test, y_train, y_test = split_data_train_test(input_df, pred_attr=pred_attr, test_size=test_size,
                                                             random_state=random_state, shuffle=shuffle, outdir=out_dir,
                                                             exclude_year=exclude_year)
    regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    regressor.fit(x_train, y_train)
    y_pred = regressor.predict(x_test)

    feature_imp = " ".join(str(np.round(i, 3)) for i in regressor.feature_importances_)
    train_score = np.round(regressor.score(x_train, y_train), 3)
    test_score = np.round(regressor.score(x_test, y_test), 3)
    mae = np.round(metrics.mean_absolute_error(y_test, y_pred), 3)
    rmse = np.round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 3)

    if plot_graphs:
        plt.plot(y_pred, y_test, 'ro')
        plt.xlabel('GW_Predict')
        plt.ylabel('GW_Actual')
        plt.show()

    df = {'N_Estimator': [n_estimators], 'Random_State': [random_state], 'F_IMP': [feature_imp],
          'Train_Score': [train_score], 'Test_Score': [test_score], 'MAE': [mae], 'RMSE': [rmse]}
    print(df)
    df = pd.DataFrame(data=df)
    df.to_csv(out_dir + 'RF_Results.csv', mode='a', index=False)
    return regressor


def create_pred_raster(rf_model, out_raster, actual_raster_dir, pred_year=2015, pred_attr='GW_KS',
                       plot_graphs=False, exclude_year=True):
    """
    Create prediction raster
    :param rf_model: Pre-built Random Forest Model
    :param out_raster: Output raster
    :param actual_raster_dir: Ground truth raster files required for prediction
    :param pred_attr: Prediction attribute name in the dataframe
    :param plot_graphs: Plot Actual vs Prediction graph
    :param exclude_year: Set true for not adding the year attribute for model prediction
    :return: None
    """

    raster_files = glob(actual_raster_dir + '*_' + str(pred_year) + '*.tif')
    raster_arr_dict = {}
    print('\nPrediction running')
    for raster_file in raster_files:
        print('Obtaining', raster_file, '...')
        sep = raster_file.rfind('_')
        variable, year = raster_file[raster_file.rfind('/') + 1: sep], raster_file[sep + 1: raster_file.rfind('.')]
        raster_arr, actual_file = rops.read_raster_as_arr(raster_file)
        raster_shape = raster_arr.shape
        raster_arr = raster_arr.reshape(raster_shape[0] * raster_shape[1])
        nan_pos = np.isnan(raster_arr)
        raster_arr[nan_pos] = 0
        raster_arr_dict[variable] = raster_arr
        raster_arr_dict['YEAR'] = [year] * raster_arr.shape[0]

    input_df = pd.DataFrame(data=raster_arr_dict)
    actual_arr = raster_arr_dict[pred_attr]
    drop_columns = [pred_attr]
    if exclude_year:
        drop_columns += ['YEAR']
    input_df = input_df.drop(columns=drop_columns)
    pred_arr = rf_model.predict(input_df)
    actual_arr[nan_pos] = actual_file.nodata
    pred_arr[nan_pos] = actual_file.nodata
    actual_values = actual_arr[actual_arr != actual_file.nodata]
    pred_values = pred_arr[pred_arr != actual_file.nodata]
    mae = np.round(metrics.mean_absolute_error(actual_values, pred_values), 3)
    r_squared = metrics.r2_score(actual_values, pred_values)
    rmse = np.round(np.sqrt(metrics.mean_squared_error(actual_values, pred_values)), 3)
    print('MAE=', mae, 'RMSE=', rmse, 'R^2=', r_squared)

    if plot_graphs:
        plt.plot(pred_values, actual_values, 'ro')
        plt.xlabel('GW_Predict')
        plt.ylabel('GW_Actual')
        plt.show()

    pred_arr = pred_arr.reshape(raster_shape)
    rops.write_raster(pred_arr, actual_file, transform=actual_file.transform, outfile_path=out_raster)
