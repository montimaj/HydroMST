from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split as tts
import pandas
from glob import glob
import numpy as np

from Python_Port import rasterops as rops


def create_dataframe(input_file_dir):
    """
    Create dataframe from file list
    :param input_file_dir: Input directory where the file names begin with <Variable>_<Year>, e.g, ET_2015.tif
    :return: Pandas dataframe
    """

    raster_dict = {}
    for f in glob(input_file_dir):
        sep = f.rfind('_')
        variable, year = f[: sep], f[sep + 1, f.rfind('.')]
        raster_arr = rops.read_raster_as_arr(f)
        raster_arr = raster_arr.reshape(raster_arr.shape[0] * raster_arr.shape[1])
        raster_dict[(variable, year)] = raster_arr

    return pandas.DataFrame(data=raster_dict)




