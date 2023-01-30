# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import sklearn.utils as sk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from glob import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance
from mpl_toolkits.mplot3d import axes3d
from Python_Files.hydrolibs import rasterops as rops
from Python_Files.hydrolibs import model_analysis as ma


def create_dataframe(input_file_dir, input_gmd_file, output_dir, column_names=None, pattern='*.tif', exclude_years=(),
                     exclude_vars=(), make_year_col=True, ordering=False, label_attr='Label', load_gmd_info=False):
    """
    Create dataframe from file list
    :param input_file_dir: Input directory where the file names begin with <Variable>_<Year>, e.g, ET_2015.tif
    :param input_gmd_file: Input GMD shape file
    :param output_dir: Output directory
    :param column_names: Dataframe column names, these must be df headers
    :param pattern: File pattern to look for in the folder
    :param exclude_years: Exclude these years from the dataframe
    :param exclude_vars: Exclude these variables from the dataframe
    :param make_year_col: Make a dataframe column entry for year
    :param ordering: Set True to order dataframe column names
    :param label_attr: Label attribute present in the shapefile
    :param load_gmd_info: Set True to load previously created GMD info raster
    :return: GMD Numpy array
    :return: Pandas dataframe
    """

    raster_file_dict = defaultdict(lambda: [])
    for f in glob(input_file_dir + pattern):
        sep = f.rfind('_')
        variable, year = f[f.rfind(os.sep) + 1: sep], f[sep + 1: f.rfind('.')]
        if variable not in exclude_vars and int(year) not in exclude_years:
            raster_file_dict[int(year)].append(f)
    raster_dict = {}
    flag = False
    years = sorted(list(raster_file_dict.keys()))
    df = None
    raster_arr = None
    gmd_arr = rops.get_gmd_info_arr(raster_file_dict[years[0]][0], input_gmd_file, output_dir=output_dir,
                                    label_attr=label_attr, load_gmd_info=load_gmd_info)
    gmd_arr = gmd_arr.ravel()
    for year in years:
        file_list = raster_file_dict[year]
        for raster_file in file_list:
            raster_arr = rops.read_raster_as_arr(raster_file, get_file=False)
            raster_arr = raster_arr.reshape(raster_arr.shape[0] * raster_arr.shape[1])
            variable = raster_file[raster_file.rfind(os.sep) + 1: raster_file.rfind('_')]
            raster_dict[variable] = raster_arr
        if make_year_col:
            raster_dict['YEAR'] = [year] * raster_arr.shape[0]
        if not flag:
            df = pd.DataFrame(data=raster_dict)
            flag = True
        else:
            df = pd.concat([df, pd.DataFrame(data=raster_dict)])
    df['GMD'] = gmd_arr.tolist() * len(years)
    df = df.dropna(axis=0)
    df = reindex_df(df, column_names=column_names, ordering=ordering)
    out_df = output_dir + 'raster_df.csv'
    df.to_csv(out_df, index=False)
    return df


def reindex_df(df, column_names, ordering=False):
    """
    Reindex dataframe columns
    :param df: Input dataframe
    :param column_names: Dataframe column names, these must be df headers
    :param ordering: Set True to apply ordering
    :return: Reindexed dataframe
    """
    if not column_names:
        column_names = df.columns
        ordering = True
    if ordering:
        column_names = sorted(column_names)
    return df.reindex(column_names, axis=1)


def get_rf_model(rf_file):
    """
    Get existing RF model object
    :param rf_file: File path to RF model
    :return: RandomForestRegressor
    """

    return pickle.load(open(rf_file, mode='rb'))


def split_data_train_test_ratio(input_df, pred_attr='GW', shuffle=True, random_state=0, test_size=0.2, outdir=None,
                                test_year=None, test_gmd=None, use_gmd=False):
    """
    Split data based on train-test percentage
    :param input_df: Input dataframe
    :param pred_attr: Prediction attribute name
    :param shuffle: Default True for shuffling
    :param random_state: Random state used during train test split
    :param test_size: Test data size percentage (0<=test_size<=1)
    :param outdir: Set path to store intermediate files
    :param test_year: Build test data from only this year
    :param test_gmd: Build test data from only this GMD, use_gmd must be set to True
    :param use_gmd: Set True to build test data from only test_gmd
    :return: X_train, X_test, y_train, y_test
    """

    years = set(input_df['YEAR'])
    gmds = set(input_df['GMD'])
    x_train_df = pd.DataFrame()
    x_test_df = pd.DataFrame()
    y_train_df = pd.DataFrame()
    y_test_df = pd.DataFrame()
    flag = False
    if (test_year in years) or (use_gmd and test_gmd in gmds):
        flag = True
    selection_var = years
    selection_label = 'YEAR'
    test_var = test_year
    if use_gmd:
        selection_var = gmds
        selection_label = 'GMD'
        test_var = test_gmd
    for svar in selection_var:
        selected_data = input_df.loc[input_df[selection_label] == svar]
        y = selected_data[pred_attr]
        x_train, x_test, y_train, y_test = train_test_split(selected_data, y, shuffle=shuffle,
                                                            random_state=random_state, test_size=test_size)
        x_train_df = x_train_df.append(x_train)
        if (flag and test_var == svar) or not flag:
            x_test_df = x_test_df.append(x_test)
            y_test_df = pd.concat([y_test_df, y_test])
        y_train_df = pd.concat([y_train_df, y_train])

    if outdir:
        x_train_df.to_csv(outdir + 'X_Train.csv', index=False)
        x_test_df.to_csv(outdir + 'X_Test.csv', index=False)
        y_train_df.to_csv(outdir + 'Y_Train.csv', index=False)
        y_test_df.to_csv(outdir + 'Y_Test.csv', index=False)

    return x_train_df, x_test_df, y_train_df[0].ravel(), y_test_df[0].ravel()


def split_data_attribute(input_df, pred_attr='GW', outdir=None, test_years=(2016, ), test_gmds=(1, 2, 3),
                         use_gmds=False, shuffle=True, random_state=0):
    """
    Split data based on a particular attribute like year or GMD
    :param input_df: Input dataframe
    :param pred_attr: Prediction attribute name
    :param outdir: Set path to store intermediate files
    :param test_years: Build test data from only these years
    :param test_gmds: Build test data from only these GMDs, use_gmds must be set to True
    :param use_gmds: Set True to build test data from only test_gmds
    :param shuffle: Set False to stop data shuffling
    :param random_state: Seed for PRNG
    :return: X_train, X_test, y_train, y_test
    """

    years = set(input_df['YEAR'])
    gmds = set(input_df['GMD'])
    x_train_df = pd.DataFrame()
    x_test_df = pd.DataFrame()
    y_train_df = pd.DataFrame()
    y_test_df = pd.DataFrame()
    selection_var = years
    selection_label = 'YEAR'
    test_vars = test_years
    if use_gmds:
        selection_var = gmds
        selection_label = 'GMD'
        test_vars = test_gmds
    for svar in selection_var:
        selected_data = input_df.loc[input_df[selection_label] == svar]
        y_t = selected_data[pred_attr]
        x_t = selected_data
        if svar not in test_vars:
            x_train_df = x_train_df.append(x_t)
            y_train_df = pd.concat([y_train_df, y_t])
        else:
            x_test_df = x_test_df.append(x_t)
            y_test_df = pd.concat([y_test_df, y_t])

    if shuffle:
        x_train_df = sk.shuffle(x_train_df, random_state=random_state)
        y_train_df = sk.shuffle(y_train_df, random_state=random_state)
        x_test_df = sk.shuffle(x_test_df, random_state=random_state)
        y_test_df = sk.shuffle(y_test_df, random_state=random_state)

    if outdir:
        x_train_df.to_csv(outdir + 'X_Train.csv', index=False)
        x_test_df.to_csv(outdir + 'X_Test.csv', index=False)
        y_train_df.to_csv(outdir + 'Y_Train.csv', index=False)
        y_test_df.to_csv(outdir + 'Y_Test.csv', index=False)

    return x_train_df, x_test_df, y_train_df[0].ravel(), y_test_df[0].ravel()


def split_data_gmd_ratio(input_df, pred_attr='GW', shuffle=True, random_state=0, test_size=0.2, outdir=None,
                         gmd_id=4):
    """
    Split data based on train-test percentage of a particular GMD
    :param input_df: Input dataframe
    :param pred_attr: Prediction attribute name
    :param shuffle: Default True for shuffling
    :param random_state: Random state used during train test split
    :param test_size: Test data size percentage (0<=test_size<=1)
    :param outdir: Set path to store intermediate files
    :param gmd_id: Build train/test data from only this GMD
    :return: X_train, X_test, y_train, y_test
    """

    gmds = input_df.GMD.unique()
    if isinstance(gmd_id, list) or isinstance(gmd_id, tuple):
        gmd_id = gmd_id[0]
    if gmd_id not in gmds:
        gmd_id = 4
    gmd_df = input_df[input_df.GMD == gmd_id]
    y = gmd_df[pred_attr].to_frame()
    x_train, x_test, y_train, y_test = train_test_split(
        gmd_df, y,
        shuffle=shuffle,
        random_state=random_state,
        test_size=test_size
    )
    if outdir:
        x_train.to_csv(outdir + 'X_Train.csv', index=False)
        x_test.to_csv(outdir + 'X_Test.csv', index=False)
        y_train.to_csv(outdir + 'Y_Train.csv', index=False)
        y_test.to_csv(outdir + 'Y_Test.csv', index=False)
    return x_train, x_test, y_train[pred_attr].ravel(), y_test[pred_attr].ravel()


def create_pdplots(x_train, rf_model, outdir, plot_3d=False, descriptive_labels=False):
    """
    Create partial dependence plots
    :param x_train: Training set
    :param rf_model: Random Forest model
    :param outdir: Output directory for storing partial dependence data
    :param plot_3d: Set True for creating pairwise 3D plots
    :param descriptive_labels: Set True to get descriptive labels
    :return: None
    """

    print('Plotting...')
    feature_names = x_train.columns.values.tolist()
    plot_labels = {'AGRI': 'AGRI', 'URBAN': 'URBAN', 'SW': 'SW', 'ET': 'ET (mm)', 'P': 'P (mm)', 'Crop': 'CC'}
    if descriptive_labels:
        plot_labels = {'AGRI': 'Agriculture density', 'URBAN': 'Urban density', 'SW': 'Surface water density',
                       'ET': 'Evapotranspiration (mm)', 'P': 'Precipitation (mm)', 'Crop': 'Crop Coefficient'}
    feature_indices = range(len(feature_names))
    feature_dict = {}
    if plot_3d:
        x_train = x_train[:500]
        for fi in feature_indices:
            for fj in feature_indices:
                feature_check = (fi != fj) and ((fi, fj) not in feature_dict.keys()) and ((fj, fi) not in
                                                                                          feature_dict.keys())
                if feature_check:
                    print(feature_names[fi], feature_names[fj])
                    feature_dict[(fi, fj)] = True
                    feature_dict[(fj, fi)] = True
                    f_pefix = outdir + 'PDP_' + feature_names[fi] + '_' + feature_names[fj]
                    saved_files = glob(outdir + '*' + feature_names[fi] + '_' + feature_names[fj] + '*')
                    if not saved_files:
                        pdp, axes = partial_dependence(rf_model, x_train, features=(fi, fj))
                        x, y = np.meshgrid(axes[0], axes[1])
                        z = pdp[0].T
                        np.save(f_pefix + '_X', x)
                        np.save(f_pefix + '_Y', y)
                        np.save(f_pefix + '_Z', z)
                    else:
                        x = np.load(f_pefix + '_X.npy')
                        y = np.load(f_pefix + '_Y.npy')
                        z = np.load(f_pefix + '_Z.npy')
                    fig = plt.figure()
                    ax = axes3d.Axes3D(fig)
                    surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='k')
                    ax.set_xlabel(plot_labels[feature_names[fi]])
                    ax.set_ylabel(plot_labels[feature_names[fj]])
                    ax.set_zlabel('GW Pumping (mm)')
                    plt.colorbar(surf, shrink=0.3, aspect=5)
                    plt.show()
    else:
        fnames = []
        for name in feature_names:
            fnames.append(plot_labels[name])
        plot_partial_dependence(rf_model, features=feature_indices, X=x_train, feature_names=fnames, n_jobs=-1)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()


def rf_regressor(input_df, out_dir, n_estimators=500, random_state=0, bootstrap=True, max_features=None, test_size=0.2,
                 pred_attr='GW', shuffle=True, plot_graphs=False, plot_3d=False, plot_dir=None, drop_attrs=(),
                 test_case='', test_year=None, test_gmd=None, use_gmd=False, split_attribute=True, load_model=True,
                 calc_perm_imp=False, split_gmd_ratio=False):
    """
    Perform random forest regression
    :param input_df: Input pandas dataframe
    :param out_dir: Output file directory for storing intermediate results
    :param n_estimators: RF hyperparameter
    :param random_state: RF hyperparameter
    :param bootstrap: RF hyperparameter
    :param max_features: RF hyperparameter
    :param test_size: Required only if split_yearly=False
    :param pred_attr: Prediction attribute name in the dataframe
    :param shuffle: Set False to stop data shuffling
    :param plot_graphs: Plot Actual vs Prediction graph
    :param plot_3d: Plot pairwise 3D partial dependence plots
    :param plot_dir: Directory for storing PDP data
    :param drop_attrs: Drop these specified attributes
    :param test_case: Used for writing the test case number to the CSV
    :param test_year: Build test data from only this year. Use tuple of years to split train and test data using
    #split_data_attribute
    :param test_gmd: Build test data from only this GMD, use_gmd must be set to True. Use tuple of years to split train
    and test data using #split_data_attribute
    :param use_gmd: Set True to build test data from only test_gmd
    :param split_attribute: Split train test data based on a particular attribute like year or GMD
    :param load_model: Load an earlier pre-trained RF model
    :param calc_perm_imp: Set True to get permutation importances on train and test data
    :param split_gmd_ratio: If True, then data is split based on the particular GMD set in test_gmd (must have a single
    GMD id, otherwise, the first GMD from the list is taken). If test_gmd is None, then GMD 4 is used as default.
    This flag if True will supersede other splitting flags.
    :return: Random forest model
    """

    saved_model = glob(out_dir + '*rf_model*')
    if load_model and saved_model:
        regressor = get_rf_model(saved_model[0])
        x_train = pd.read_csv(out_dir + 'X_Train.csv')
        y_train = pd.read_csv(out_dir + 'Y_Train.csv')
        x_test = pd.read_csv(out_dir + 'X_Test.csv')
        y_test = pd.read_csv(out_dir + 'Y_Test.csv')
    else:
        if split_gmd_ratio:
            x_train, x_test, y_train, y_test = split_data_gmd_ratio(
                input_df, pred_attr=pred_attr,
                test_size=test_size,
                random_state=random_state, shuffle=shuffle,
                outdir=out_dir, gmd_id=test_gmd
            )
        else:
            if not split_attribute:
                x_train, x_test, y_train, y_test = split_data_train_test_ratio(
                    input_df, pred_attr=pred_attr,
                    test_size=test_size,
                    random_state=random_state, shuffle=shuffle,
                    outdir=out_dir, test_year=test_year,
                    test_gmd=test_gmd, use_gmd=use_gmd
                )
            else:
                x_train, x_test, y_train, y_test = split_data_attribute(
                    input_df, pred_attr=pred_attr,
                    outdir=out_dir, test_years=test_year,
                    shuffle=shuffle, random_state=random_state,
                    test_gmds=test_gmd, use_gmds=use_gmd
                )
        drop_columns = [pred_attr] + list(drop_attrs)
        x_train = x_train.drop(columns=drop_columns)
        x_test = x_test.drop(columns=drop_columns)
        regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, bootstrap=bootstrap,
                                          max_features=3, n_jobs=-1, oob_score=True, min_samples_leaf=8,
                                          max_depth=16)
        regressor.fit(x_train, y_train)
        pickle.dump(regressor, open(out_dir + 'rf_model', mode='wb'))

    print('Predictor... ')
    y_pred = regressor.predict(x_test)
    feature_imp = " ".join(str(np.round(i, 2)) for i in regressor.feature_importances_)
    permutation_imp_train, permutation_imp_test = None, None
    if calc_perm_imp:
        permutation_imp_train = permutation_importance(regressor, x_train, y_train, n_repeats=10, n_jobs=-1,
                                                       random_state=random_state)
        permutation_imp_train = " ".join(str(np.round(i, 2)) for i in permutation_imp_train.importances_mean)
        permutation_imp_test = permutation_importance(regressor, x_test, y_test, n_repeats=10, n_jobs=-1,
                                                      random_state=random_state)
        permutation_imp_test = " ".join(str(np.round(i, 2)) for i in permutation_imp_test.importances_mean)
    train_score = np.round(regressor.score(x_train, y_train), 2)
    test_score = np.round(regressor.score(x_test, y_test), 2)
    r2_score, standard_r2, mae, rmse, nmae, nrmse = ma.get_error_stats(y_test, y_pred)
    oob_score = np.round(regressor.oob_score_, 2)
    print(x_train.columns)
    df = {'Test': [test_case], 'N_Estimator': [n_estimators], 'MF': [max_features], 'F_IMP': [feature_imp],
          'Train_Score': [train_score], 'Test_Score': [test_score], 'OOB_Score': [oob_score], 'R2': [r2_score],
          'Standard R2': [standard_r2], 'MAE': [mae], 'RMSE': [rmse], 'NMAE': [nmae], 'NRMSE': [nrmse]}
    if calc_perm_imp:
        df['P_IMP_TRAIN'], df['P_IMP_TEST'] = [permutation_imp_train], [permutation_imp_test]
    print('Model statistics:', df)
    df = pd.DataFrame(data=df)
    with open(out_dir + 'RF_Results.csv', 'a') as f:
        df.to_csv(f, mode='a', index=False, header=f.tell() == 0)
    if plot_graphs:
        create_pdplots(x_train=x_train, rf_model=regressor, outdir=plot_dir, plot_3d=plot_3d)
    return regressor


def create_pred_raster(rf_model, out_raster, actual_raster_dir, column_names=None, pred_year=2015, pred_attr='GW',
                       drop_attrs=(), only_pred=False, calculate_errors=True, ordering=False):
    """
    Create prediction raster
    :param rf_model: Pre-built Random Forest Model
    :param out_raster: Output raster
    :param actual_raster_dir: Ground truth raster files required for prediction
    :param column_names: Dataframe column names, these must be df headers
    :param pred_year: Prediction year
    :param pred_attr: Prediction attribute name in the dataframe
    :param drop_attrs: Drop these specified attributes (Must be exactly the same as used in rf_regressor module)
    :param only_pred: Set True to disable raster creation and for showing only the error metrics,
    automatically set to False if calculate_errors is False
    :param calculate_errors: Calculate error metrics if actual observations are present
    :param ordering: Set True to order dataframe column names
    :return: Tuple containing R2, Standard R2, MAE, RMSE, NMAE, and NRMSE (rounded to 2 decimal places)
    """

    raster_files = glob(actual_raster_dir + '*_' + str(pred_year) + '*.tif')
    raster_arr_dict = {}
    nan_pos_dict = {}
    actual_file = None
    raster_shape = None
    for raster_file in raster_files:
        sep = raster_file.rfind('_')
        variable, year = raster_file[raster_file.rfind(os.sep) + 1: sep], raster_file[sep + 1: raster_file.rfind('.')]
        raster_arr, actual_file = rops.read_raster_as_arr(raster_file)
        raster_shape = raster_arr.shape
        raster_arr = raster_arr.reshape(raster_shape[0] * raster_shape[1])
        nan_pos_dict[variable] = np.isnan(raster_arr)
        if not only_pred:
            raster_arr[nan_pos_dict[variable]] = 0
        raster_arr_dict[variable] = raster_arr
        raster_arr_dict['YEAR'] = [year] * raster_arr.shape[0]
    input_df = pd.DataFrame(data=raster_arr_dict)
    input_df = input_df.dropna(axis=0)
    input_df = reindex_df(input_df, column_names=column_names, ordering=ordering)
    drop_columns = [pred_attr] + list(drop_attrs)
    if not calculate_errors:
        drop_cols = drop_columns
        if not column_names:
            drop_cols.remove(pred_attr)
        input_df = input_df.drop(columns=drop_cols)
        pred_arr = rf_model.predict(input_df)
        if not only_pred:
            for nan_pos in nan_pos_dict.values():
                pred_arr[nan_pos] = actual_file.nodata
        r2_score, standard_r2, mae, rmse, nmae, nrmse = (np.nan, ) * 6
    else:
        if only_pred:
            actual_arr = input_df[pred_attr]
        else:
            actual_arr = raster_arr_dict[pred_attr]
        input_df = input_df.drop(columns=drop_columns)
        pred_arr = rf_model.predict(input_df)
        if not only_pred:
            for nan_pos in nan_pos_dict.values():
                actual_arr[nan_pos] = actual_file.nodata
                pred_arr[nan_pos] = actual_file.nodata
            actual_values = actual_arr[actual_arr != actual_file.nodata]
            pred_values = pred_arr[pred_arr != actual_file.nodata]
        else:
            actual_values = actual_arr
            pred_values = pred_arr
        r2_score, standard_r2, mae, rmse, nmae, nrmse = ma.get_error_stats(actual_values, pred_values)
    if not only_pred:
        pred_arr = pred_arr.reshape(raster_shape)
        rops.write_raster(pred_arr, actual_file, transform=actual_file.transform, outfile_path=out_raster)
    return r2_score, standard_r2, mae, rmse, nmae, nrmse


def predict_rasters(rf_model, actual_raster_dir, out_dir, pred_years, column_names=None, drop_attrs=(), pred_attr='GW',
                    only_pred=False, exclude_years=(2019, ), ordering=False):
    """
    Create prediction rasters from input data
    :param rf_model: Pre-trained Random Forest Model
    :param actual_raster_dir: Directory containing input rasters
    :param out_dir: Output directory for predicted rasters
    :param pred_years: Tuple containing prediction years
    :param column_names: Dataframe column names, these must be df headers
    :param drop_attrs: Drop these specified attributes (Must be exactly the same as used in rf_regressor module)
    :param pred_attr: Prediction Attribute
    :param only_pred: Set true to disable raster creation and for showing only the error metrics
    :param exclude_years: Exclude these years from error analysis, only the respective predicted rasters are generated
    :param ordering: Set True to order dataframe column names
    :return: None
    """

    for pred_year in pred_years:
        out_pred_raster = out_dir + 'pred_' + str(pred_year) + '.tif'
        calculate_errors = True
        if pred_year in exclude_years:
            calculate_errors = False
        r2_score, standard_r2, mae, rmse, nmae, nrmse = create_pred_raster(rf_model, out_raster=out_pred_raster,
                                                                           actual_raster_dir=actual_raster_dir,
                                                                           pred_year=pred_year, drop_attrs=drop_attrs,
                                                                           pred_attr=pred_attr, only_pred=only_pred,
                                                                           calculate_errors=calculate_errors,
                                                                           column_names=column_names, ordering=ordering)
        print('YEAR', pred_year, ': R2 = ', r2_score, 'SR2 = ', standard_r2, 'MAE =', mae, 'RMSE =', rmse,
              'NMAE =', nmae, 'NRMSE =', nrmse)
