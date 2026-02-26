import os
import zipfile
import logging.config

import helpers_general, helpers_logging
import helpers_foresight, foresight_logging


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class ETLError(Exception):
    """ Called for 'expected' errors within the ETL process """


def setup_foresight_logger(config_dict, print_to_console=False):
    """ Sets up main logger that will log general progress

    Parameters
    ----------
    config_dict : dict
        Dictionary of configuration options read from configuration yaml file

    print_to_console : bool
        Whether or not to print logging messages to the console

    Returns
    -------
    Logger
        Logging object that can be used for logging messages
    """
    module = "transform"
    additional_suffixes = [config_dict["general"]["data_source"]]
    log_dir = config_dict["general"]["log_dir"]
    log_level = helpers_logging.get_logging_level(config_dict["general"]["log_level"])
    foresight_logger = foresight_logging.ForesightLogging(module, log_dir, log_level, print_to_console,
                                                          additional_suffixes=additional_suffixes)
    return foresight_logger


def is_etl_stage_required(etl_config, etl_stage, etl_stage_file):
    """ Determines whether we need to do the specified ETL stage based on existence of given file and overwrite
    preference

    Parameters
    ----------
    etl_config : ETLConfig
        Custom class of configuration options

    etl_stage : str
        "extract" or "transform

    etl_stage_file : str
        The target file to look for to determine if stage has been previously completed

    Returns
    -------
    bool
        Whether given etl stage is required or not
    """
    if etl_stage == "extract":
        overwrite = False
    elif etl_stage == "transform":
        overwrite = etl_config.overwrite_transformed
    else:
        raise ETLError(f"Unknown etl stage provided: {etl_stage}")

    try:
        helpers_general.readable_file(etl_stage_file)
        if overwrite:
            LOGGER.warning(f"Overwrite specified and existing file found. "
                           f"Carrying out {etl_stage} ETL stage and overwriting existing file {etl_stage_file}")
            etl_stage_required = True
        else:
            LOGGER.info(f"Existing file found at {etl_stage_file}. Skipping {etl_stage}...")
            etl_stage_required = False

    except (IOError, ValueError):
        # file doesn't exist so need to process!
        etl_stage_required = True

    return etl_stage_required


def build_transformed_file_path(etl_config, etl_stage_description=None, override_datetime=None, region=None):
    """ Builds the path to the file name that the transformed dataset will be saved to

    Parameters
    ----------
    etl_config : ETLConfig
        Custom class of configuration options

    etl_stage_description : str, optional
        If provided, adds an ETL stage description to add to the final directory structure

    override_datetime : datetime, optional
        If provided, will use this datetime instead of the one in etl_config to build file path

    Returns
    -------
    str
        Full path to the transformed file
    """
    if override_datetime is not None:
        dt = override_datetime
    else:
        dt = etl_config.etl_datetime

    netcdf_output_dir = helpers_foresight.build_etl_dir(etl_config.load_directory, etl_config.data_source,
                                                        "transformed", additional_dirs=etl_stage_description,
                                                        file_date=dt)
    helpers_general.check_path_exists_and_create(netcdf_output_dir)

    netcdf_file = helpers_foresight.build_transformed_etl_file(etl_config.data_source, dt, region=region)
    netcdf_full_path = os.path.join(netcdf_output_dir, netcdf_file)

    return netcdf_full_path


def unzip_file(local_zip_file, del_zip=False):
    with zipfile.ZipFile(local_zip_file, "r") as zip_extractor:
        LOGGER.info("Extracting files from zip archive")
        extract_dir = helpers_general.check_file_output_dir(local_zip_file)
        zip_extractor.extractall(extract_dir)

    if del_zip:
        # delete original zip
        helpers_general.check_file_and_delete(local_zip_file)

    return extract_dir
