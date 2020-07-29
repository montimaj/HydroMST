# Author: Sayantan Majumdar
# Email: smxnv@mst.edu

import os
from shutil import copy2
from glob import glob


def make_gdal_sys_call_str(gdal_path, gdal_command, args, verbose=True):
    """
    Make GDAL system call string
    :param gdal_path: GDAL directory path, in Windows replace with OSGeo4W directory path, e.g. '/usr/bin/gdal/' on
    Linux or Mac and 'C:/OSGeo4W64/' on Windows, the '/' at the end is mandatory
    :param gdal_command: GDAL command to use
    :param args: GDAL arguments as a list
    :param verbose: Set True to print system call info
    :return: GDAL system call string,
    """

    sys_call = [gdal_path + gdal_command] + args
    if os.name == 'nt':
        gdal_path += 'OSGeo4W.bat'
        sys_call = [gdal_path] + [gdal_command] + args
    if verbose:
        print(sys_call)
    return sys_call


def makedirs(directory_list):
    """
    Create directory for storing files
    :param directory_list: List of directories to create
    :return: None
    """

    for directory_name in directory_list:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)


def make_proper_dir_name(dir_str):
    """
    Append os.sep to dir if not present
    :param dir_str: Directory path
    :return: Corrected directory path
    """

    if dir_str is None:
        return None
    sep = [os.sep, '/']
    if dir_str[-1] not in sep:
        return dir_str + os.sep
    return dir_str


def copy_file(input_file, output_file, suffix='', ext='.tif', verbose=True):
    """
    Copy a single file
    :param input_file: Input file name
    :param output_file: Output file name (should not contain extension)
    :param suffix: Suffix string to append to output_file
    :param ext: Extension of output file
    :param verbose: Set True to get info on copy
    :return: None
    """

    output_file += suffix + ext
    if verbose:
        print('Copying', input_file, 'to', output_file, '...')
    copy2(input_file, output_file)


def copy_files(input_dir_list, target_dir, pattern_list, year_list, rep=False, verbose=True):
    """
    Copy files from input directories to target directory
    :param input_dir_list: List of input directories, the files should be named as <var>_* _<year>.
    :param target_dir: Target directory, the output files are named as <var>_<year>
    :param year_list: List of years to copy
    :param rep: Replicate files for each year (this is useful to copying land use files which doesn't change much over
    time)
    :param pattern_list: File pattern list ordered according to input_dir_list
    :param verbose: Set True to get info on copy
    :return: None
    """

    for input_dir, pattern in zip(input_dir_list, pattern_list):
        file_list = glob(input_dir + pattern)
        for f in file_list:
            file_name = f[f.rfind(os.sep) + 1:]
            outfile = target_dir + file_name[: file_name.rfind('_') + 1]
            ext_sep = file_name.rfind('.')
            ext = file_name[ext_sep:]
            if not rep:
                year = file_name[file_name.rfind('_') + 1: ext_sep]
                if int(year) in year_list:
                    copy_file(f, outfile, suffix=year, ext=ext, verbose=verbose)
            else:
                for year in year_list:
                    copy_file(f, outfile, suffix=str(year), ext=ext, verbose=verbose)
