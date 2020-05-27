# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import pandas as pd
from Python_Files.hydrolibs import rasterops as rops
from Python_Files.hydrolibs import vectorops as vops
from Python_Files.hydrolibs.sysops import makedirs, make_proper_dir_name, copy_files
from Python_Files.hydrolibs import random_forest_regressor as rfr
from Python_Files.hydrolibs import model_analysis as ma
from glob import glob


class HydroML:

    def __init__(self, input_dir, file_dir, output_dir, input_ts_dir, output_shp_dir, output_gw_raster_dir,
                 input_gmd_file, input_state_file, input_cdl_file, gdal_path):
        """
        Constructor for initializing class variables
        :param input_dir: Input data directory
        :param file_dir: Directory for storing intermediate files
        :param output_dir: Output directory
        :param input_ts_dir: Input directory containing the time series data
        :param output_shp_dir: Output shapefile directory
        :param output_gw_raster_dir: Output GW raster directory
        :param input_gmd_file: Input GMD file
        :param input_state_file: Input state shapefile
        :param input_cdl_file: Input NASS CDL file path
        :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
        Linux or Mac and 'C:/OSGeo4W64/' on Windows
        """

        self.input_dir = make_proper_dir_name(input_dir)
        self.file_dir = make_proper_dir_name(file_dir)
        self.output_dir = make_proper_dir_name(output_dir)
        self.input_ts_dir = make_proper_dir_name(input_ts_dir)
        self.output_shp_dir = make_proper_dir_name(output_shp_dir)
        self.output_gw_raster_dir = make_proper_dir_name(output_gw_raster_dir)
        self.gdal_path = make_proper_dir_name(gdal_path)
        self.input_gmd_file = input_gmd_file
        self.input_state_file = input_state_file
        self.input_cdl_file = input_cdl_file
        self.input_gmd_reproj_file = None
        self.input_state_reproj_file = None
        self.final_gw_dir = None
        self.ref_raster = None
        self.raster_reproj_dir = None
        self.reclass_reproj_file = None
        self.raster_mask_dir = None
        self.land_use_dir_list = None
        self.rf_data_dir = None
        makedirs([self.output_dir, self.output_gw_raster_dir, self.output_shp_dir])

    def extract_shp_from_gdb(self, input_gdb_dir, year_list, attr_name='AF_USED', already_extracted=False):
        """
        Extract shapefiles from geodatabase (GDB)
        :param input_gdb_dir: Input GDB directory
        :param year_list: List of years to extract
        :param attr_name: Attribute name for shapefile
        :param already_extracted: Set True to disable extraction
        :return: None
        """

        if not already_extracted:
            print('Extracting GW data from GDB...')
            vops.extract_gdb_data(input_gdb_dir, attr_name=attr_name, year_list=year_list, outdir=self.output_shp_dir)
        else:
            print("GW shapefiles already extracted")

    def reproject_shapefiles(self, already_reprojected=False):
        """
        Reproject GMD and state shapefiles
        :param already_reprojected: Set True to disable reprojection
        :return: None
        """

        gmd_reproj_dir = make_proper_dir_name(self.file_dir + 'gmds/reproj')
        state_reproj_dir = make_proper_dir_name(self.file_dir + 'state/reproj')
        self.input_gmd_reproj_file = gmd_reproj_dir + 'input_gmd_reproj.shp'
        self.input_state_reproj_file = state_reproj_dir + 'input_state_reproj.shp'
        if not already_reprojected:
            makedirs([gmd_reproj_dir, state_reproj_dir])
            ref_shp = glob(self.output_shp_dir + '*.shp')[0]
            vops.reproject_vector(self.input_gmd_file, outfile_path=self.input_gmd_reproj_file, ref_file=ref_shp,
                                  raster=False)
            vops.reproject_vector(self.input_state_file, outfile_path=self.input_state_reproj_file, ref_file=ref_shp,
                                  raster=False)
        else:
            print('GMD and State shapefiles are already reprojected')

    def clip_gw_shpfiles(self, new_clip_file=None, already_clipped=False, extent_clip=True):
        """
        Clip GW shapefiles based on GMD extent
        :param new_clip_file: Input clip file for clipping GW shapefiles (e.g, it could be a watershed shapefile),
        required only if you don't want to clip using GMD extent. Should be in the same projection system
        :param already_clipped: Set False to re-clip shapefiles
        :param extent_clip: Set False to clip by cutline, if shapefile consists of multiple polygons, then this won't
        work
        :return: None
        """

        clip_file = self.input_gmd_reproj_file
        if new_clip_file:
            clip_file = new_clip_file
        clip_shp_dir = make_proper_dir_name(self.output_shp_dir + 'Clipped')
        if not already_clipped:
            print('Clipping GW shapefiles...')
            makedirs([clip_shp_dir])
            vops.clip_vectors(self.output_shp_dir, clip_file=clip_file, outdir=clip_shp_dir, gdal_path=self.gdal_path,
                              extent_clip=extent_clip)
        else:
            print('GW Shapefiles already clipped')
        self.output_shp_dir = clip_shp_dir

    def create_gw_rasters(self, xres=5000, yres=5000, max_gw=1000, raster_mask=None, crop_rasters=False, ext_mask=True,
                          convert_units=True, already_created=True):
        """
        Create GW rasters from shapefiles
        :param xres: X-Resolution (map unit)
        :param yres: Y-Resolution (map unit)
        :param max_gw: Maximum GW pumping in mm. Any value higher than this will be set to no data
        :param raster_mask: Raster mask (shapefile) for cropping raster, required only if crop_rasters=True
        :param crop_rasters: Set False to disable raster cropping
        :param ext_mask: Set True to crop by cutline, if shapefile consists of multiple polygons, then this won't
        work
        :param convert_units: If true, converts GW pumping values in acreft to mm
        :param already_created: Set False to re-compute GW pumping rasters
        :return: None
        """

        fixed_dir = make_proper_dir_name(self.output_gw_raster_dir + 'Fixed')
        cropped_dir = make_proper_dir_name(self.output_gw_raster_dir + 'Cropped')
        converted_dir = make_proper_dir_name(self.output_gw_raster_dir + 'Converted')
        if not already_created:
            print('Converting SHP to TIF...')
            makedirs([fixed_dir])
            vops.shps2rasters(self.output_shp_dir, self.output_gw_raster_dir, xres=xres, yres=yres, smoothing=0,
                              gdal_path=self.gdal_path, gridding=False)
            if convert_units:
                max_gw *= xres * yres / (1.233 * 1e+6)
            rops.fix_large_values(self.output_gw_raster_dir, max_threshold=max_gw, outdir=fixed_dir)
            if crop_rasters:
                makedirs([cropped_dir])
                if not raster_mask:
                    raster_mask = self.input_state_reproj_file
                rops.crop_rasters(fixed_dir, outdir=cropped_dir, input_mask_file=raster_mask, ext_mask=ext_mask,
                                  gdal_path=self.gdal_path)
            if convert_units:
                print('Changing GW units from acreft to mm')
                makedirs([converted_dir])
                input_dir = fixed_dir
                if crop_rasters:
                    input_dir = cropped_dir
                rops.convert_gw_data(input_dir, converted_dir)
        else:
            print('GW  pumping rasters already created')
        if convert_units:
            self.final_gw_dir = converted_dir
        elif crop_rasters:
            self.final_gw_dir = cropped_dir
        else:
            self.final_gw_dir = fixed_dir

    def reclassify_cdl(self, reclass_dict, pattern='*.tif', already_reclassified=False):
        """
        Reclassify raster
        :param reclass_dict: Dictionary where key values are tuples representing the interval for reclassification, the
        dictionary values represent the new class
        :param pattern: File pattern required for reprojection
        :param already_reclassified: Set True to disable reclassification
        :return: None
        """

        reclass_dir = make_proper_dir_name(self.file_dir + 'Reclass')
        self.reclass_reproj_file = reclass_dir + 'reclass_reproj.tif'
        self.ref_raster = glob(self.final_gw_dir + pattern)[0]
        if not already_reclassified:
            print('Reclassifying CDL 2015 data...')
            makedirs([reclass_dir])
            reclass_file = reclass_dir + 'reclass.tif'
            rops.reclassify_raster(self.input_cdl_file, reclass_dict, reclass_file)
            rops.reproject_raster(reclass_file, self.reclass_reproj_file, from_raster=self.ref_raster,
                                  gdal_path=self.gdal_path)
            rops.filter_nans(self.reclass_reproj_file, ref_file=self.ref_raster, outfile_path=self.reclass_reproj_file)
        else:
            print('Already reclassified')

    def reproject_rasters(self, pattern='*.tif', already_reprojected=False):
        """
        Reproject rasters based on GW as reference raster
        :param pattern: File pattern to look for
        :param already_reprojected: Set True to disable raster reprojection
        :return: None
        """

        self.raster_reproj_dir = make_proper_dir_name(self.file_dir + 'Reproj_Rasters')
        if not already_reprojected:
            print('Reprojecting rasters...')
            makedirs([self.raster_reproj_dir])
            rops.reproject_rasters(self.input_ts_dir, ref_raster=self.ref_raster, outdir=self.raster_reproj_dir,
                                   pattern=pattern, gdal_path=self.gdal_path)
        else:
            print('All rasters already reprojected')

    def mask_rasters(self, pattern='*.tif', already_masked=False):
        """
        Mask rasters based on reference GW raster
        :param pattern: File pattern to look for
        :param already_masked: Set True to disable raster masking
        :return: None
        """

        self.raster_mask_dir = make_proper_dir_name(self.file_dir + 'Masked_Rasters')
        if not already_masked:
            print('Masking rasters...')
            makedirs([self.raster_mask_dir])
            rops.mask_rasters(self.raster_reproj_dir, ref_raster=self.ref_raster, outdir=self.raster_mask_dir,
                              pattern=pattern)
        else:
            print('All rasters already masked')

    def create_land_use_rasters(self, class_values=(1, 2, 3), class_labels=('AGRI', 'SW', 'URBAN'),
                                smoothing_factors=(3, 5, 3), already_created=False):
        """
        Create land use rasters from the reclassified raster
        :param class_values: List of land use class values to consider for creating separate rasters
        :param class_labels: List of class_labels ordered according to land_uses
        :param smoothing_factors: Smoothing factor (sigma value for Gaussian filter) to use while smoothing
        :param already_created: Set True to disable land use raster generation
        :return: None
        """

        self.land_use_dir_list = [make_proper_dir_name(self.file_dir + class_label) for class_label in class_labels]
        if not already_created:
            for idx, (class_value, class_label) in enumerate(zip(class_values, class_labels)):
                print('Extracting land use raster for', class_label, '...')
                raster_dir = self.land_use_dir_list[idx]
                makedirs([raster_dir])
                raster_file = raster_dir + class_label + '.tif'
                rops.apply_raster_filter2(self.reclass_reproj_file, outfile_path=raster_file, val=class_value)
                raster_masked = raster_dir + class_label + '_masked.tif'
                rops.filter_nans(raster_file, self.ref_raster, outfile_path=raster_masked)
                raster_flt = raster_dir + class_label + '_flt.tif'
                rops.apply_gaussian_filter(raster_masked, ref_file=self.ref_raster, outfile_path=raster_flt,
                                           sigma=smoothing_factors[idx], normalize=True, ignore_nan=False)
        else:
            print('Land use rasters already created')

    def create_dataframe(self, year_list, column_names=None, ordering=False, load_df=False, exclude_years=(2019, ),
                         verbose=True):
        """
        Create dataframe from preprocessed files
        :param year_list: List of years for which the dataframe will be created
        :param column_names: Dataframe column names, these must be df headers
        :param ordering: Set True to order dataframe column names
        :param load_df: Set true to load existing dataframe
        :param exclude_years: List of years to exclude from dataframe
        :param verbose: Get extra information if set to True
        :return: Pandas dataframe object
        """

        self.rf_data_dir = make_proper_dir_name(self.file_dir + 'RF_Data')
        df_file = self.output_dir + 'raster_df.csv'
        if load_df:
            print('Getting dataframe...')
            return pd.read_csv(df_file)
        else:
            print('Copying files...')
            makedirs([self.rf_data_dir])
            input_dir_list = [self.final_gw_dir] + [self.raster_mask_dir]
            pattern_list = ['*.tif'] * len(input_dir_list)
            copy_files(input_dir_list, target_dir=self.rf_data_dir, year_list=year_list, pattern_list=pattern_list,
                       verbose=verbose)
            pattern_list = ['*_flt.tif'] * len(self.land_use_dir_list)
            copy_files(self.land_use_dir_list, target_dir=self.rf_data_dir, year_list=year_list,
                       pattern_list=pattern_list, rep=True, verbose=verbose)
            print('Creating dataframe...')
            df = rfr.create_dataframe(self.rf_data_dir, out_df=df_file, column_names=column_names, make_year_col=True,
                                      exclude_years=exclude_years, ordering=ordering)
            return df

    def tune_parameters(self, df, pred_attr, drop_attrs=()):
        """
        Tune random forest hyperparameters
        :param df: Input pandas dataframe object
        :param pred_attr: Target attribute
        :param drop_attrs: List of attributes to drop from the df
        :return: None
        """

        n_features = len(df.columns) - len(drop_attrs) - 1
        test_cases = [2014, 2012, range(2012, 2017)]
        est_range = range(100, 601, 100)
        f_range = range(1, n_features + 1)
        ts = []
        for n in range(1, len(test_cases) + 1):
            ts.append('T' + str(n))
        for ne in est_range:
            for nf in f_range:
                for y, t in zip(test_cases, ts):
                    if not isinstance(y, range):
                        ty = (y,)
                    else:
                        ty = y
                    rfr.rf_regressor(df, self.output_dir, n_estimators=ne, random_state=0, pred_attr=pred_attr,
                                     drop_attrs=drop_attrs, test_year=ty, shuffle=False, plot_graphs=False,
                                     split_yearly=True, bootstrap=True, max_features=nf, test_case=t)

    def build_model(self, df, n_estimators=100, random_state=0, bootstrap=True, max_features=3, test_size=None,
                    pred_attr='GW', shuffle=False, plot_graphs=False, plot_3d=False, drop_attrs=(), test_year=(2012,),
                    split_yearly=True, load_model=False):
        """
        Build random forest model
        :param df: Input pandas dataframe object
        :param pred_attr: Target attribute
        :param drop_attrs: List of attributes to drop from the df
        :param n_estimators: RF hyperparameter
        :param random_state: RF hyperparameter
        :param bootstrap: RF hyperparameter
        :param max_features: RF hyperparameter
        :param test_size: Required only if split_yearly=False
        :param pred_attr: Prediction attribute name in the dataframe
        :param shuffle: Set False to stop data shuffling
        :param plot_graphs: Plot Actual vs Prediction graph
        :param plot_3d: Plot pairwise 3D partial dependence plots
        :param drop_attrs: Drop these specified attributes
        :param test_year: Build test data from only this year(s).
        :param split_yearly: Split train test data based on years
        :param load_model: Load an earlier pre-trained RF model
        :return: Fitted RandomForestRegressor object
        """

        print('Building RF Model...')
        plot_dir = make_proper_dir_name(self.output_dir + 'Partial_Plots/PDP_Data')
        makedirs([plot_dir])
        rf_model = rfr.rf_regressor(df, self.output_dir, n_estimators=n_estimators, random_state=random_state,
                                    pred_attr=pred_attr, drop_attrs=drop_attrs, test_year=test_year, shuffle=shuffle,
                                    plot_graphs=plot_graphs, plot_3d=plot_3d, split_yearly=split_yearly,
                                    bootstrap=bootstrap, plot_dir=plot_dir, max_features=max_features,
                                    load_model=load_model, test_size=test_size)
        return rf_model

    def get_predictions(self, rf_model, pred_years, column_names=None, final_mask=None, ordering=False, pred_attr='GW',
                        only_pred=False, exclude_years=(2019,), drop_attrs=(), crop_rasters=False):
        """
        Get prediction results and/or rasters
        :param rf_model: Fitted RandomForestRegressor model
        :param pred_years: Predict for these years
        :param column_names: Dataframe column names, these must be df headers
        :param final_mask: Raster mask required to properly clip the actual and predicted rasters, required only if
        crop_rasters=True
        :param ordering: Set True to order dataframe column names
        :param pred_attr: Prediction attribute name in the dataframe
        :param only_pred: Set True to disable prediction raster generation
        :param exclude_years: List of years to exclude from dataframe
        :param drop_attrs: Drop these specified attributes
        :param crop_rasters: Set True to crop actual and predicted rasters
        :return: Predicted raster directory path
        """

        print('Predicting...')
        pred_out_dir = make_proper_dir_name(self.output_dir + 'Predicted_Rasters')
        makedirs([pred_out_dir])
        rfr.predict_rasters(rf_model, pred_years=pred_years, drop_attrs=drop_attrs, out_dir=pred_out_dir,
                            actual_raster_dir=self.rf_data_dir, pred_attr=pred_attr, only_pred=only_pred,
                            exclude_years=exclude_years, column_names=column_names, ordering=ordering)
        if crop_rasters:
            crop_dir = make_proper_dir_name(pred_out_dir + 'Cropped_Rasters')
            makedirs([crop_dir])
            pattern = pred_attr + '*.tif'
            if not final_mask:
                final_mask = self.input_state_reproj_file
            rops.crop_multiple_rasters(self.rf_data_dir, outdir=crop_dir, input_shp_file=final_mask, pattern=pattern)
            pattern = 'pred*.tif'
            rops.crop_multiple_rasters(pred_out_dir, outdir=crop_dir, input_shp_file=final_mask, pattern=pattern)
            pred_out_dir = crop_dir
        return pred_out_dir


def run_gw(analyze_only=False, load_files=True, load_rf_model=False, use_gmds=True):
    """
    Main function for running the project, some variables require to be hardcoded
    :param analyze_only: Set True to just produce analysis results, all required files must be present
    :param load_files: Set True to load existing files, needed only if analyze_only=False
    :param load_rf_model: Set True to load existing Random Forest model, needed only if analyze_only=False
    :param use_gmds: Set False to use entire GW raster for analysis
    :return: None
    """

    gee_data = ['Apr_Sept/', 'Apr_Aug/', 'Annual/']
    input_dir = '../Inputs/Data/Kansas_GW/'
    file_dir = '../Inputs/Files_' + gee_data[0]
    output_dir = '../Outputs/Output_' + gee_data[0]
    input_ts_dir = input_dir + 'GEE_Data_' + gee_data[0]
    output_shp_dir = file_dir + 'GW_Shapefiles/'
    output_gw_raster_dir = file_dir + 'GW_Rasters/'
    input_gmd_file = input_dir + 'gmds/ks_gmds.shp'
    input_cdl_file = input_dir + 'CDL/CDL_KS_2015.tif'
    input_gdb_dir = input_dir + 'ks_pd_data_updated2018.gdb'
    input_state_file = input_dir + 'Kansas/kansas.shp'
    gdal_path = 'C:/OSGeo4W64/'
    gw_dir = file_dir + 'RF_Data/'
    pred_gw_dir = output_dir + 'Predicted_Rasters/'
    grace_csv = input_dir + 'GRACE/TWS_GRACE.csv'
    ks_class_dict = {(0, 59.5): 1,
                     (66.5, 77.5): 1,
                     (203.5, 255): 1,
                     (110.5, 111.5): 2,
                     (111.5, 112.5): 0,
                     (120.5, 124.5): 3,
                     (59.5, 61.5): 0,
                     (130.5, 195.5): 0
                     }
    drop_attrs = ('YEAR',)
    pred_attr = 'GW'
    if not analyze_only:
        gw = HydroML(input_dir, file_dir, output_dir, input_ts_dir, output_shp_dir, output_gw_raster_dir,
                     input_gmd_file, input_state_file, input_cdl_file, gdal_path)
        gw.extract_shp_from_gdb(input_gdb_dir, year_list=range(2002, 2019), already_extracted=load_files)
        gw.reproject_shapefiles(already_reprojected=load_files)
        gw.create_gw_rasters(already_created=load_files)
        gw.reclassify_cdl(ks_class_dict, already_reclassified=load_files)
        gw.reproject_rasters(already_reprojected=load_files)
        gw.mask_rasters(already_masked=load_files)
        gw.create_land_use_rasters(already_created=load_files)
        df = gw.create_dataframe(year_list=range(2002, 2020), load_df=load_files)
        rf_model = gw.build_model(df, test_year=range(2011, 2019), drop_attrs=drop_attrs, pred_attr=pred_attr,
                                  load_model=load_rf_model, max_features=5, plot_graphs=False)
        pred_gw_dir = gw.get_predictions(rf_model=rf_model, pred_years=range(2002, 2020), drop_attrs=drop_attrs,
                                         pred_attr=pred_attr, only_pred=False)
    ma.run_analysis(gw_dir, pred_gw_dir, grace_csv, use_gmds=use_gmds, input_gmd_file=input_gmd_file,
                    out_dir=output_dir)


run_gw(analyze_only=True, load_files=True, load_rf_model=False, use_gmds=True)
