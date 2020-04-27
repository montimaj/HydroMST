from Python_Files.hydrolibs import rasterops as rops
from Python_Files.hydrolibs.sysops import makedirs
from glob import glob
from Python_Files.hydrolibs import random_forest_regressor as rfr
import os


outdir = '../Middle_East/Output_ME/Data/'
makedirs([outdir])
# input_raster_dir = '../Middle_East/Data/'
# ref_raster = glob(input_raster_dir + 'ET*.tif')[0]
# rops.reproject_rasters(input_raster_dir, ref_raster=ref_raster, outdir=outdir, pattern='SW*.tif')
# rops.reclassify_raster2(input_raster_dir + 'AGRI_2016.tif', {(1, 2): 1, (3, 255): 0},
#                         outfile_path=outdir + 'AGRI_2016.tif')
# rops.reclassify_raster2(input_raster_dir + 'AGRI_2016.tif', {(0, 255): 0}, outfile_path=outdir + 'URBAN_2016.tif')

rf_model = rfr.get_rf_model('../Output/rf_model')
pred_years = [2016]
drop_attrs = ['YEAR']
pred_out_dir = outdir + 'Results/'
makedirs([pred_out_dir])
pred_attr = 'GW'
rfr.predict_rasters(rf_model, pred_years=pred_years, drop_attrs=drop_attrs, out_dir=pred_out_dir,
                    actual_raster_dir=outdir, pred_attr=pred_attr, only_pred=False)
