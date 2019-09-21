from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas
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
            df = pandas.DataFrame(data=raster_dict)
            flag = True
        else:
            df = df.append(pandas.DataFrame(data=raster_dict))

    df = df.dropna(axis=0)
    df.to_csv(out_df, index=False)
    return df


def rf_regressor(input_df, out_dir, n_estimators=200, random_state=0, test_size=0.2, pred_attr='GW_KS', shuffle=True,
                 plot_graphs=False):
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
    :return: Random forest model
    """

    y = input_df[pred_attr]
    dataset = input_df.drop(columns=[pred_attr])
    X = dataset.iloc[:, 0: len(dataset.columns)].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        shuffle=shuffle)
    regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    feature_imp = " ".join(str(np.round(i, 3)) for i in regressor.feature_importances_)
    train_score = np.round(regressor.score(X_train, y_train), 3)
    test_score = np.round(regressor.score(X_test, y_test), 3)
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
    df = pandas.DataFrame(data=df)
    df.to_csv(out_dir + 'RF_Results.csv', mode='a', index=False)
    return regressor


def create_pred_raster(rf_model, input_df, out_raster, actual_raster_file, pred_attr='GW_KS', plot_graphs=False):
    """
    Create prediction raster
    :param rf_model: Pre-built Random Forest Model
    :param input_df: Input pandas dataframe for prediction
    :param out_raster: Output raster
    :param actual_raster_file: Ground truth raster file of the predicted variable for creating predicted raster
    :param pred_attr: Prediction attribute name in the dataframe
    :param plot_graphs: Plot Actual vs Prediction graph
    :return: None
    """

    # actual_arr, actual_file = rops.read_raster_as_arr(actual_raster_file)
    actual_arr = input_df[pred_attr]
    input_df = input_df.drop(columns=[pred_attr])
    pred_arr = rf_model.predict(input_df)
    mae = np.round(metrics.mean_absolute_error(actual_arr, pred_arr), 3)
    rmse = np.round(np.sqrt(metrics.mean_squared_error(actual_arr, pred_arr)), 3)
    r_squared = metrics.r2_score(actual_arr, pred_arr)
    print('MAE=', mae, 'RMSE=', rmse, 'R^2=', r_squared)

    if plot_graphs:
        plt.plot(pred_arr, actual_arr, 'ro')
        plt.xlabel('GW_Predict')
        plt.ylabel('GW_Actual')
        plt.show()

    # out_arr = np.full_like(ref_arr, fill_value=0)


