import os
import enum
import re
import datetime
import json
import logging
import glob

from dateutil.relativedelta import relativedelta

import jba_common.helpers_general as helpers_general
from jba_common.foresight.entities import CoreCSV

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

OUTPUT_TYPES = {
    "timeseries": "ts",
}

FILE_TYPES = {
    "map_list": ".txt",
    "calc_list": ".txt",
    "csv": ".csv",
    "vrt": ".vrt",
    "tif": ".tif",
    "shp": ".shp",
    "map": ".map",
    "impacts": ".xlsx"
}

PRODUCTS = {
    "fluvial": "for",
    "monitoring": "mon",
    "heavy_rainfall": "hvr",
    "surface_water": "sfw"
}


@enum.unique
class JobState(enum.IntEnum):
    Requested = 1
    ECS_Task_Created = 2
    Job_Started = 3
    Job_Finished_With_Success = 4
    Job_Finished_With_Error = 5


def build_core_out_path(base_out_path, product_name, map_source, data_source, project_name, output_type, file_type,
                        run_datetime=None, forecast_start_datetime=None, forecast_datetime=None, ensemble=None,
                        time_window=None, filename_suffix=None, datetime_format=None, add_hour_dir=False,
                        must_exist=False):
    """ Builds the full out path to the core output file, with directory and filename

    Parameters
    ----------
    base_out_path : str
        Path to the top level of the core output, must exist

    product_name : str
        Foresight product name, forecasting or monitoring

    map_source : str
        The source of the flood depth maps used in preprocessing, e.g. jba for JBA Risk maps

    project_name : str
        Short project name that preprocessing out path relates to - will contain locale and optional descriptor,
        e.g. esw_2019_5m = 2019 vintage maps, at 5m resolution, for the locale of England Scotland Wales

    output_type : str
        e.g. timeseries

    file_type : str
        e.g. csv, tif, shp

    run_datetime : datetime.datetime, optional
        Datetime object at which the outputs are valid. Sub directories will be created using the year, month and day

    forecast_start_datetime : datetime.datetime, optional
        If in forecasting mode, datetime object of the forecast start date.  Can be omitted if the forecast period
        is at regular time intervals, or if forecast is for an instantaneous instant in time.  Could be used for
        forecasts that cover a time window, e.g. the maximum forecast value over a three-hour window.

    forecast_datetime : datetime.datetime, optional
        If in forecasting mode, datetime object of the forecast date

    ensemble : int, optional
        If using ensembles, the ensemble value of the output

    time_window : int, optional
        If using time window, the time window value of the output

    filename_suffix : str, optional
        Any additional file name part to be added to the end of the filename

    datetime_format : str, optional
        The strftime format that datetime string should display in the filename.  Defaults to "%Y%m%dT%H%MZ"
        if None given

    add_hour_dir : bool, False
        Whether or not the out path should include a directory for the hour.  Default is false.

    must_exist : bool, False
        If True, the directory will not be created if doesn't exist, and the full path must already exist or an
        error is raised

    Returns
    -------
    str
        Full output path to the core output file to save
    """

    # build the directory
    out_dir = build_core_out_dir(base_out_path, product_name, map_source, data_source, project_name, output_type, file_type,
                                 run_datetime=run_datetime, add_hour_dir=add_hour_dir, must_exist=must_exist)
    # build the filename
    out_file = build_core_out_filename(product_name, project_name, output_type, file_type,
                                       run_datetime=run_datetime, forecast_start_datetime=forecast_start_datetime,
                                       forecast_datetime=forecast_datetime, ensemble=ensemble, time_window=time_window,
                                       filename_suffix=filename_suffix, datetime_format=datetime_format)

    # join together
    out_path = os.path.join(out_dir, out_file)

    if must_exist:
        helpers_general.readable_file(out_path)

    return out_path


def build_core_out_dir(base_out_path, product_name, map_source, data_source, project_name, output_type, file_type,
                       run_datetime=None, add_hour_dir=False, must_exist=False):
    """ Builds the output path where core output files will be saved

    Parameters
    ----------
    base_out_path : str
        Path to the top level of the core output, must exist

    product_name : str
        Foresight product name, forecasting or monitoring

    map_source : str
        The source of the flood depth maps used in preprocessing, e.g. jba for JBA Risk maps

    project_name : str
        Short project name that preprocessing out path relates to - will contain locale and optional descriptor,
        e.g. esw_2019_5m = 2019 vintage maps, at 5m resolution, for the locale of England Scotland Wales

    output_type : str
        e.g. timeseries

    file_type : str
        e.g. csv, tif, shp

    run_datetime : datetime.datetime, optional
        Datetime object at which the outputs are valid. Sub directories will be created using the year, month and day

    add_hour_dir : bool, False
        Whether or not the out path should include a directory for the hour.  Default is false.

    must_exist : bool, False
        If True, the directory will not be created if doesn't exist, and the path must already exist or an
        error is raised

    Returns
    -------
    str
        Full path to the core output path
    """

    helpers_general.writeable_dir(base_out_path)
    out_path = os.path.join(base_out_path,
                            product_name.lower(),
                            map_source.lower(),
                            data_source.lower(),
                            project_name.lower(),
                            output_type.lower(),
                            file_type.lower())

    if run_datetime is not None:
        out_path = os.path.join(out_path,
                                f"{run_datetime.year:04d}",
                                f"{run_datetime.month:02d}",
                                f"{run_datetime.day:02d}")

        if add_hour_dir:
            out_path = os.path.join(out_path, f"{run_datetime.hour:02d}")

    if must_exist:
        helpers_general.writeable_dir(out_path)
    else:
        helpers_general.check_path_exists_and_create(out_path)

    return out_path


def build_core_out_filename(product_name, project_name, output_type, file_type, run_datetime=None,
                            forecast_start_datetime=None, forecast_datetime=None, ensemble=None, time_window=None,
                            filename_suffix=None, datetime_format=None):
    """ Builds the output filename that core output files will be saved as

    Parameters
    ----------
    product_name : str
        Foresight product name, forecasting or monitoring

    project_name : str
        Short project name that preprocessing out path relates to - will contain locale and optional descriptor,
        e.g. esw_2019_5m = 2019 vintage maps, at 5m resolution, for the locale of England Scotland Wales

    output_type : str
        e.g. timeseries

    file_type : str
        e.g. csv, tif, shp

    run_datetime : datetime.datetime, optional
        Datetime object at which the outputs are valid. Sub directories will be created using the year, month and day

    forecast_start_datetime : datetime.datetime, optional
        If in forecasting mode, datetime object of the forecast start date.  Can be omitted if the forecast period
        is at regular time intervals, or if forecast is for an instantaneous instant in time.  Could be used for
        forecasts that cover a time window, e.g. the maximum forecast value over a three-hour window.

    forecast_datetime : datetime.datetime, optional
        If in forecasting mode, datetime object of the forecast end date

    ensemble : int, optional
        If using ensembles, the ensemble value of the output

    time_window : int, optional
        If using time window, the time window value of the output

    filename_suffix : str, optional
        Any additional file name part to be added to the end of the filename

    datetime_format : str, optional
        The strftime format that datetime string should display in the filename. Defaults to "%Y%m%dT%H%MZ" if None
        given

    Returns
    -------
    str
        Filename of the core output
    """

    # build the filename in parts:
    # description
    try:
        if output_type.startswith("timeseries"):
            output_type_short = "ts"
        else:
            output_type_short = OUTPUT_TYPES[output_type]
    except KeyError:
        raise ValueError(f"Unexpected output type {output_type} when building output filename")

    try:
        product_abbr = PRODUCTS[product_name]
    except KeyError:
        raise ValueError(f"Unexpected product name {product_name} when building output filename")

    filename_part_description = f"{product_abbr}_{project_name}_{output_type_short}".lower()

    # date time stamps
    if datetime_format is None:
        datetime_format = "%Y%m%dT%H%MZ"

    filename_part_forecast_date = ""
    if forecast_datetime is not None:  # may be none if building a filename without dates, e.g. mapfile template
        if isinstance(forecast_datetime, datetime.datetime):
            filename_part_forecast_date = f"_fe{forecast_datetime.strftime(datetime_format)}"
        else:
            filename_part_forecast_date = f"_fe{forecast_datetime}"

    filename_part_forecast_start_date = ""
    if forecast_start_datetime is not None:  # may be none if building a filename without dates, e.g. mapfile template
        if isinstance(forecast_start_datetime, datetime.datetime):
            filename_part_forecast_start_date = f"_fs{forecast_start_datetime.strftime(datetime_format)}"
        else:
            filename_part_forecast_start_date = f"_fs{forecast_start_datetime}"

    filename_part_run_date = ""
    if run_datetime is not None:  # may be none if building a filename without dates, e.g. mapfile template
        if isinstance(run_datetime, datetime.datetime):
            filename_part_run_date = f"_rd{run_datetime.strftime(datetime_format)}"
        else:
            filename_part_run_date = f"_rd{run_datetime}"

    filename_part_ensemble = ""
    if ensemble is not None:
        filename_part_ensemble = f"_ens{ensemble:02d}"

    filename_part_time_window = ""
    if time_window is not None and time_window != 0:  # 0 indicates the normal latest value, so no windowing
        filename_part_time_window = f"_{time_window:02d}day_max"

    filename_part_suffix = ""
    if filename_suffix is not None:
        filename_part_suffix = f"_{filename_suffix}"

    # final filename
    try:
        extension = FILE_TYPES[file_type]
        map_list_filename = (f"{filename_part_description}"
                             f"{filename_part_time_window}"
                             f"{filename_part_forecast_date}"
                             f"{filename_part_forecast_start_date}"
                             f"{filename_part_run_date}"
                             f"{filename_part_ensemble}"
                             f"{filename_part_suffix}"
                             f"{extension}")

    except KeyError:
        raise ValueError(f"Unexpected file type {file_type} when building output filename")

    return map_list_filename


def build_core_file_regex(file_type, full_path=False):
    """ Builds a regular expression (regex) pattern that can be used to match output files generated by Foresight.

    Parameters
    ----------
    file_type : str
        e.g. csv, tif, shp

    full_path : bool, optional
        Whether or not to build regex that matches directory structure as well as the filename

    Returns
    -------
    re.Pattern
        The compiled regex pattern that can be used to match/search strings
    """
    # Adjust project pattern to include numbers
    project_pattern = "(?P<project>[a-zA-Z0-9]+_?[a-zA-Z0-9]*)"  # Allow numbers in project part, e.g., arg5
    description_pattern = f"(?P<product>[a-zA-Z]+)_{project_pattern}_(?P<output_type_short>[a-zA-Z]+)"

    # optional time window part
    time_window_pattern = "(?:_(?P<time_window>[0-9]{2}[a-zA-Z_]+))?"

    # optional forecast date
    forecast_start_date_pattern = "(?:_fs(?P<forecast_start_date_time>[0-9]{8}T[0-9]{4}Z))?"

    # optional forecast date
    forecast_date_pattern = "(?:_fe(?P<forecast_date_time>[0-9]{8}T[0-9]{4}Z))?"
    # optional run date
    run_date_pattern = "(?:_rd(?P<run_date_time>[0-9]{8}T[0-9]{4}Z))?"
    # optional ensemble
    ensemble_pattern = "(?:_ens(?P<ensemble>[0-9]{2}))?"
    # optional additional suffix
    suffix_pattern = "(?:_(?P<suffix>.*))?"

    # File extension
    extension = FILE_TYPES[file_type]

    # Full filename pattern
    pattern = (rf"{description_pattern}{time_window_pattern}{forecast_date_pattern}{forecast_start_date_pattern}"
               rf"{run_date_pattern}{ensemble_pattern}{suffix_pattern}{extension}")

    # Handle the full path case
    if full_path:
        slash_pattern = r"[\\/]*"
        product_long_pattern = "(?P<product_long>[a-zA-Z_]+)"
        map_source_pattern = "(?P<map_source>[a-zA-Z_]+)"
        project_pattern = "[a-zA-Z0-9_]+"  # No need to capture as it should be contained in filename
        output_type_long_pattern = "(?P<output_type_long>[a-zA-Z_]+)"

        run_date_dir_pattern = rf"(?:[0-9]{{4}}{slash_pattern}[0-9]{{2}}{slash_pattern}[0-9]{{2}}{slash_pattern})?"

        dir_pattern = (rf"foresight{slash_pattern}"
                       rf"output{slash_pattern}"
                       rf"{product_long_pattern}{slash_pattern}"
                       rf"{map_source_pattern}{slash_pattern}"
                       rf"{project_pattern}{slash_pattern}"
                       rf"{output_type_long_pattern}{slash_pattern}"
                       rf"{file_type}{slash_pattern}"
                       rf"{run_date_dir_pattern}")

        pattern = f"{dir_pattern}{pattern}"

    # Compile and return the pattern
    pattern_compiled = re.compile(pattern)
    return pattern_compiled


def build_preprocessing_out_path(base_out_path, map_source, project_name, preprocessing_stage=None, iz_id=None,
                                 create=True):
    """ Builds the output path where preprocessing files will be saved

    Parameters
    ----------
    base_out_path : str
        Path to the top level of the preprocessing output

    map_source : str
        The source of the flood depth maps used in preprocessing, e.g. jba for JBA Risk maps

    project_name : str
        Short project name that preprocessing out path relates to - will contain locale and optional descriptor,
        e.g. esw_2019_5m = 2019 vintage maps, at 5m resolution, for the locale of England Scotland Wales

    preprocessing_stage : str, optional
        Stage of the preprocessing process to build output path for, one of "clipped", "interpolated",
        "range_classified", "volume_lookups".  If None, then won't use.

    iz_id : int, optional
        The id of the impact zone to process. Optional for volume_lookups

    create : bool, optional
        Whether to create the directory if not found

    Returns
    -------
    str
        Full path to the output directory to save into
    """

    if preprocessing_stage is not None:
        if not preprocessing_stage.lower() in ["clipped", "interpolated", "range_classified", "volume_lookups", "logs"]:
            raise ValueError(f"Unknown preprocessing stage: {preprocessing_stage}")

    out_path = os.path.join(base_out_path,
                            map_source.lower(),
                            project_name.lower())

    if preprocessing_stage is not None:
        out_path = os.path.join(out_path, preprocessing_stage.lower())

    if iz_id is not None:
        iz_group_dir = get_impact_zone_group_dir(iz_id)
        out_path = os.path.join(out_path,
                                iz_group_dir,
                                f"{iz_id}")

    if create:
        helpers_general.check_path_exists_and_create(out_path)

    return out_path


def get_impact_zone_group_dir(iz_id):
    """ Gets the group directory that impact zone outputs will be saved to.

    Parameters
    ----------
    iz_id : int
        The impact zone ID to get the group directory for

    Returns
    -------
    str
        Group directory name
    """

    divisor = 1000
    lower_range_value = (iz_id // divisor) * divisor
    upper_range_value = lower_range_value + (divisor - 1)
    iz_group_dir = f"{lower_range_value}_{upper_range_value}"
    return iz_group_dir


def get_impact_zone_file_string(iz_id):
    """ Returns the padded string representation of impact zone id for use in file names etc.

    Parameters
    ----------
    iz_id : int
        The impact zone ID to get padded file string for

    Returns
    -------
    str
        Padded string representation of impact zone id
    """
    iz_id_pad = 7  # how many digits to pad the iz id out to
    padded_iz_str = f"IZ{iz_id:0{iz_id_pad}d}"
    return padded_iz_str


def build_etl_dir(base_etl_path, discharge_source, raw_or_transformed, additional_dirs=None, file_date=None,
                  raw_date_format="Y-M-D", read_only=False):
    """ Builds the directory of where to find input ETL files

    Parameters
    ----------
    base_etl_path : str
        Path to the top level of the ETL output, must exist

    discharge_source : str
        The source of input data, e.g. glofas

    raw_or_transformed : str
        What type of ETL files will be in the directory, "raw" or "transformed"

    additional_dirs : str or tuple, optional
        Either a single string or tuple of strings to append to the directory structure, after raw or transformed,
        before date folders

    file_date : datetime.date, optional
        Date that the file relates to.  If omitted, will return the directory before the date stamps

    raw_date_format : str, optional
        If this is for a 'raw' path, we might need to override the default dir structure for the date part.
        Default is:  path/to/files/YEAR/MONTH/DAY/file.nc
        raw_date_format should be a string, with Y M D specified in desired order, separated by dash -
        If None or empty string provided, it will omit the date sub directories entirely.

    read_only : bool, optional
        If True, check whether the base_etl_path is read-only, otherwise check if it is writable

    Returns
    -------
    str
        Full path to the directory of etl input data
    """

    if raw_or_transformed.lower() not in ["raw", "transformed"]:
        raise ValueError(f"Unrecognised folder type in build_etl_dir {raw_or_transformed}")
    if read_only:
        helpers_general.readable_dir(base_etl_path)
    else:
        helpers_general.writeable_dir(base_etl_path)

    etl_dir = os.path.join(base_etl_path,
                           discharge_source,
                           raw_or_transformed.lower())

    if additional_dirs:
        if isinstance(additional_dirs, str):
            etl_dir = os.path.join(etl_dir, additional_dirs)
        elif isinstance(additional_dirs, tuple) or isinstance(additional_dirs, list):
            for d in additional_dirs:
                etl_dir = os.path.join(etl_dir, d)

    if file_date is not None:
        if raw_or_transformed.lower() == "raw":
            if raw_date_format:
                date_sub_dir_order = raw_date_format.split("-")
                for date_sub_dir in date_sub_dir_order:
                    if date_sub_dir.upper() == "Y":
                        etl_dir = os.path.join(etl_dir, f"{file_date.year:04d}")
                    elif date_sub_dir.upper() == "M":
                        etl_dir = os.path.join(etl_dir, f"{file_date.month:02d}")
                    elif date_sub_dir.upper() == "D":
                        etl_dir = os.path.join(etl_dir, f"{file_date.day:02d}")
                    else:
                        raise ValueError(f"Unrecognised sub directory provided for raw_date_format: {raw_date_format}. "
                                         f"Must contain only Y M or D, separated by a dash -")
        else:
            etl_dir = os.path.join(etl_dir,
                                   f"{file_date.year:04d}",
                                   f"{file_date.month:02d}",
                                   f"{file_date.day:02d}")
    return etl_dir


def build_transformed_etl_file(discharge_source, file_date, region=None):
    """" Builds the transformed etl filename with the expected Foresight format

    Parameters
    ----------
    discharge_source : str
        The source of input data, e.g. glofas

    file_date : datetime.datetime
        Datetime that the file relates to

    region : str, optional
        optional region name to assign to the filename

    Returns
    -------
    str
        Full path of expected transformed ETL file
    """
    file_date_str = file_date.strftime("%Y%m%dT%H%MZ")

    if region is not None:
        region = f"_{region}"
    else:
        region = ""

    transformed_etl_file = f"foresight_{discharge_source}{region}_rd{file_date_str}.nc"
    return transformed_etl_file


def get_latest_transformed_discharge_file(base_etl_path, discharge_source):
    """ Gets the latest available transformed input data file for given discharge source.

    Parameters
    ----------
    base_etl_path : str
        Path to the top level of the ETL output, must exist

    discharge_source : str
        The source of input data, e.g. glofas

    Returns
    -------
    str
        Path to the latest available input data file for this discharge source
    """

    etl_dir = build_etl_dir(base_etl_path, discharge_source, "transformed")
    helpers_general.readable_dir(etl_dir)

    # get all the files available
    etl_files = []
    for root, _, files in os.walk(etl_dir):
        z_files = [file for file in files if '_init' not in file]
        for f in z_files:
            etl_files.append(os.path.join(root, f))

    # sort the file list descending. the file naming convention should mean that the latest file is at the top of
    # this sorted list
    etl_files.sort(reverse=True)
    latest_etl_file = etl_files[0]

    # is this file still being written to? (ie. still going through the ETL process?)
    still_processing = helpers_general.is_file_being_written(latest_etl_file)
    if still_processing:
        # if so, return the 'next latest' file
        latest_etl_file = etl_files[1]

    return latest_etl_file


def get_date_of_transformed_discharge_file(transformed_file):
    """ Gets the datetime info of the given Foresight transformed input discharge data.

    Parameters
    ----------
    transformed_file : str
        Path to the Foresight transformed ETL discharge file

    Returns
    -------
    datetime.datetime
        Date info extracted from the filename
    """

    etl_file_pattern_compiled = build_discharge_netcdf_regex()
    pattern_matches = etl_file_pattern_compiled.search(transformed_file)
    file_year = int(pattern_matches["year"])
    file_month = int(pattern_matches["month"])
    file_day = int(pattern_matches["day"])
    file_hour = int(pattern_matches["hour"])
    file_minute = int(pattern_matches["minute"])

    etl_datetime = datetime.datetime(year=file_year, month=file_month, day=file_day,
                                     hour=file_hour, minute=file_minute)
    return etl_datetime


def build_discharge_netcdf_regex(netcdf_date=None, region=None):
    """ Builds a regular expression (regex) pattern that can be used to match discharge netcdf files that have been
    transformed into the common netcdf format expected by Flood Foresight

    Should conform to the file format specified in build_transformed_etl_file

    Parameters
    ----------
    netcdf_date : datetime.datetime, optional
        Date for which to build regex for

    region : str
        Optional region name held in the filename

    Returns
    -------
    re.Pattern
        The compiled regex pattern that can be used to match/search strings
    """
    # build the 'groups' that will be present in the filename

    # allow ONE underscore in the source name, e.g. glofas_global, glofas_v3p1
    source_pattern = r"(?P<source>[a-zA-Z\d]+(?:[_,-][a-zA-Z\d]+)?)"
    # optional region in the filename
    if region is not None:
        region_pattern = f"_(?P<region>{region})"
    else:
        region_pattern = "(?:_(?P<region>[a-zA-Z]+))?"

    if netcdf_date is not None:
        year_pattern = f"(?P<year>{netcdf_date.year:04d})"
        month_pattern = f"(?P<month>{netcdf_date.month:02d})"
        day_pattern = f"(?P<day>{netcdf_date.day:02d})"
    else:
        year_pattern = "(?P<year>[0-9]{4})"
        month_pattern = "(?P<month>[0-9]{2})"
        day_pattern = "(?P<day>[0-9]{2})"

    hour_pattern = "(?P<hour>[0-9]{2})"
    minute_pattern = "(?P<minute>[0-9]{2})"

    pattern = (fr"foresight_{source_pattern}{region_pattern}"
               fr"_rd{year_pattern}{month_pattern}{day_pattern}T{hour_pattern}{minute_pattern}Z.nc")

    pattern_compiled = re.compile(pattern)
    return pattern_compiled


def get_ecs_task_id():
    """ Attemps to get ECS task ID, if the current process is running in AWS.  If not, returns None
    """
    try:
        # first get the ecs metadata json file, which is where we can get the task ID:
        ecs_container_metadata_file = os.environ["ECS_CONTAINER_METADATA_FILE"]
        helpers_general.readable_file(ecs_container_metadata_file)
        # read the json file
        with open(ecs_container_metadata_file, "r") as json_file:
            ecs_metadata = json.load(json_file)

        # extract the task ID from the ARN
        ecs_task_arn = ecs_metadata["TaskARN"]
        ecs_task_id = ecs_task_arn.split("/")[-1]

    except (KeyError, IOError):
        ecs_task_id = None

    return ecs_task_id


def get_core_csv_files(csv_base_dir, product, map_source, data_source, project, run_datetime, output_type="timeseries",
                       forecast_datetime=None, ensemble=None):
    """ Retrieves all the required core CSV files based on the given options
    """
    # get the csv directory for the given date and project info
    # Added output_type="timeseries" as a key argument to allow users to specify a custom subfolder,
    # while maintaining backward compatibility with existing function calls.
    file_type = "csv"
    csv_dir = build_core_out_dir(csv_base_dir, product, map_source, data_source, project, output_type, file_type,
                                 run_datetime=run_datetime, must_exist=True)
    regex = build_core_file_regex(file_type)

    csv_dir_files = sorted(os.listdir(csv_dir))
    core_csv_files = []
    for csv_file in csv_dir_files:
        if regex.match(csv_file):
            core_csv = CoreCSV(os.path.join(csv_dir, csv_file))

            filter_match = all((
                forecast_datetime is None or core_csv.forecast_date_time == forecast_datetime,
                ensemble is None or core_csv.ensemble == f"{ensemble:02d}"
            ))

            if filter_match:
                core_csv_files.append(core_csv)
        else:
            LOGGER.warning(f"{csv_file} not recognised as a Core CSV filename. Skipping.")

    num_csv_files = len(core_csv_files)
    if num_csv_files == 0:
        raise IOError(f"No Core output CSV files found for run date {run_datetime}")

    return core_csv_files


def delete_old_downloaded_files(etl_config):
    """ Deletes GloFAS data from FTP after processing

    Parameters
    ----------
    etl_config : etl config class
        obj
    """
    input_dir = etl_config.extract_directory
    data_source = etl_config.data_source
    deletion_policy = etl_config.data_deletion_policy
    try:
        # Delete raw/transformed files (older than specified in the config)
        for data_type in ['raw', 'transformed']:
            delta = deletion_policy[data_type]
            if deletion_policy[data_type]:
                LOGGER.info(f"Deleting transformed {etl_config.data_source} files older than {delta} months")
                data_dir = os.path.join(input_dir, data_source, data_type)
                files = [f for f in glob.iglob(f'{data_dir}/**/*', recursive=True)
                         if os.path.isfile(f) and data_type in f and ('.nc' in f or '.grib2' in f)]

                older_files = [f for f in files if file_older_than(f, relativedelta(months=delta),
                                                                   deletion_policy["use_data_time"])]
                num_files = 0
                if older_files:
                    num_files = remove_file_trees_older_than(older_files, delta, deletion_policy["use_data_time"])

                LOGGER.info(f"{num_files} {data_type} {etl_config.data_source} files older than {delta} months deleted")

    except Exception as e:
        LOGGER.error(f"Error deleting files. Reason: {e}")


def get_date_age_of_file(file_str):
    reg = "\S+\/transformed\/([\S\/]+)\/[\S_]+\.nc"
    file = file_str.replace('\\', '/')
    date_str = re.match(reg, file).groups()[0]
    this_date = datetime.datetime.strptime(date_str, "%Y/%m/%d")
    date_age_of_days = (datetime.datetime.now() - this_date).days

    return date_age_of_days


def file_older_than(file, delta, use_data_time):
    """Determine if file is older than the cutoff time

    Parameters
    ----------
    file : str list
        list of file strings
    delta : relatively
        relative datetime
    use_data_time : logical
        date str
    """

    cutoff = datetime.datetime.utcnow() - delta
    if use_data_time:
        data_date, fpath = get_file_path_date(file)
        data_time = datetime.datetime.strptime(data_date, '%Y/%m/%d')
    else:
        data_time = datetime.datetime.utcfromtimestamp(os.path.getmtime(file))

    if data_time < cutoff:
        return True
    return False


def get_file_path_date(file):

    reg = re.compile(r"\S+(\d{4}/\d{2}/\d{2})\S+")

    file_path = file.replace('\\', '/')
    base_dir = re.split(r"\d{4}", file_path)[0]
    data_date = re.match(reg, file_path).groups()[0]
    fpath = os.path.join(base_dir, data_date)

    return data_date, fpath


def delete_empty_folders(output_dir):
    """delete empty folder but not recursively
    Parameters
    ----------
    output_dir : str
        output folder
    """

    try:
        # managing empty fodlers
        log_path = list(find_empty_dirs(root_dir=output_dir))
        num_folder = len(log_path)
        if log_path:
            for sub_dir in log_path:
                os.rmdir(sub_dir)
            LOGGER.info(f"Deleting {num_folder} empty folders")
        else:
            LOGGER.info(f"No empty folders to be deleted")

    except Exception as e:
        LOGGER.error(f"Error deleting empty fodlers. Reason: {e}")


def find_empty_dirs(root_dir='.'):
    for dirpath, dirs, files in os.walk(root_dir):
        if not dirs and not files:
            yield dirpath


def remove_file_trees_older_than(files, delta, use_data_time, exclude_str=''):
    """Deletes files from FTP after they are older than the retention period

    Parameters
    ----------
    files : list
        str list of files
    use_data_time : logical
        True or False
    delta : deletion_policy_regions : int
        Number of month to retain output files
    exclude_str: str
        exclude files containing the str
    """

    if exclude_str:
        all_files = [f for f in files
                     if os.path.isfile(f) and exclude_str not in f and 'zip' not in f]
    else:
        all_files = files

    old_files = [get_top_path_of_file_older_than(f, delta, use_data_time) for f in all_files
                 if get_top_path_of_file_older_than(f, delta, use_data_time) is not None]
    deleting_path = list(set(old_files))

    if deleting_path:
        for this_dir in deleting_path:
            helpers_general.check_path_and_remove_tree(this_dir)

    num_files = len(old_files)

    return num_files


def get_top_path_of_file_older_than(file, delta, use_data_time):
    """Determine if file is older than the cutoff time

    Parameters
    ----------
    file : str list
        list of file strings
    delta : int
        Number of month to retain output files
    use_data_time : logical
        True or False
    """
    data_date_fpath = get_file_path_date(file)

    if file_older_than(file, relativedelta(months=delta), use_data_time):
        return data_date_fpath[1]

    return None


def get_data_date_from_two_mark(marker1, marker2, string):
    # Find the index of the first marker
    start = string.find(marker1)

    # Find the index of the second marker
    end = string.rfind(marker2)

    # Use slicing to extract the substring between the markers
    substring = string[start + len(marker1):end]
    datetime_object = datetime.datetime.strptime(substring, '%Y%m%dT%H%M')
    date_str = datetime.datetime.strftime(datetime_object, '%Y-%m-%d-%H:%M')

    return date_str


def get_file_latest_time(config, file_type):
    file_dir = os.path.join(config.output_base_dir, config.product_name, 'jba',
                            config.project_name, 'timeseries', file_type)
    file_dates = list()
    for root, dirs, files in os.walk(file_dir):
        file_dates += [get_data_date_from_two_mark('_rd', f"Z.{file_type}", f) for f in files if f"Z.{file_type}" in f]

    latest_date = max(file_dates)
    latest_date = datetime.datetime.strptime(latest_date, '%Y-%m-%d-%H:%M')

    return latest_date


def get_src_latest_time(top_data_path):

    etl_dir = os.path.join(top_data_path, 'transformed',
                           'time_lagged_ensembles')
    file_dates = list()
    for root, dirs, files in os.walk(etl_dir):
        file_dates += [get_data_date_from_two_mark('_rd', 'Z.nc', f) for f in files if "Z.nc" in f]

    latest_date = max(file_dates)
    latest_date = datetime.datetime.strptime(latest_date, '%Y-%m-%d-%H:%M')

    return latest_date


def get_src_time(top_data_path, run_date, end_run_date):

    etl_dir = os.path.join(top_data_path, 'transformed',
                           'time_lagged_ensembles')
    file_date = datetime.datetime.strptime(run_date, '%Y-%m-%d')

    etl_dir = os.path.join(etl_dir,
                           f"{file_date.year:04d}",
                           f"{file_date.month:02d}",
                           f"{file_date.day:02d}")
    files = glob.glob(f"{etl_dir}/*.nc")
    run_date = None
    if len(files) > 0:
        date_strs = [get_data_date_from_two_mark("_rd", "Z.nc", file) for file in files]
        run_date = min(date_strs)
        if end_run_date is None:
            end_run_date = max(date_strs)

    return run_date, end_run_date
