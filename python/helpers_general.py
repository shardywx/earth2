""" General helpers
---------------------------------------------------------------------------------------------------
A module contain various miscellaneous functions that are commonly used
across the Flood Foresight pre-processing scripts.
"""

import os
import pathlib
import argparse
import datetime
import subprocess
import shutil
import time
import uuid
import zipfile
import logging
import string
import itertools
from functools import wraps, partial

import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def get_file_size(file_name, units="bytes"):
    """ Gets the file size of given file

    Parameters
    ----------
    file_name : str
        Full path to file to check file size of

    units : str
        What units to report in: bytes, kb, mb, gb, tb

    Returns
    -------
    float
        File size
    """
    file_size_units = {
        "bytes": 1,
        "kb": 1000,
        "mb": 1000000,
        "gb": 1000000000,
        "tb": 1000000000000
    }

    unit_conversion = file_size_units.get(units, 1)

    readable_file(file_name)
    check_file_size = os.stat(file_name).st_size

    check_file_size = check_file_size / unit_conversion

    return check_file_size


def get_file_matches(search_directory, regex_compiled_pattern, one_file=False):
    """ Gets all files in a given directory that match a given regular expression pattern

    Parameters
    ----------
    search_directory : str
        Path to the directory to search

    regex_compiled_pattern : re.Pattern
        The compiled regex pattern that can be used to match/search strings

    one_file : bool, optional
        Whether exactly one file should be found.  Default is False.

    Returns
    -------
    list
        A list of the full path of all the files that matched the regex in the folder
    """
    # get files in search directory, and match to the regex, adding to a list if regex matches
    search_files = os.listdir(search_directory)
    file_matches = [os.path.join(search_directory, f) for f in search_files if regex_compiled_pattern.search(f)]
    num_file_matches = len(file_matches)

    if num_file_matches == 0:
        raise FileNotFoundError(f"No files matches found in directory {search_directory} "
                                f"with given regex pattern {regex_compiled_pattern.pattern}")

    if one_file:
        if num_file_matches > 1:
            raise IOError(f"More than one file match found in directory {search_directory} "
                          f"with given regex pattern {regex_compiled_pattern.pattern}")

    return file_matches


def check_time(time_to_check, expected_format):
    """ Checks that given time string conforms to the specified format.

    Parameters
    ----------
    time_to_check : str
        String representation of time to check

    expected_format : str
        String format that the date should conform to, e.g %Y-%m-%d for 2019-11-30
        see https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes for options
    """
    checked_time = check_date(time_to_check, expected_format)
    checked_time = datetime.time(hour=checked_time.hour,
                                 minute=checked_time.minute,
                                 second=checked_time.second)
    return checked_time


def check_date(date_to_check, expected_format, set_to_midnight=False, date_only=False):
    """ Checks that given date string conforms to the specified format.
    Optional to set the time of the date to midnight.

    Parameters
    ----------
    date_to_check : str
        String representation of date to check

    expected_format : str
        String format that the date should conform to, e.g %Y-%m-%d for 2019-11-30
        see https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes for options

    set_to_midnight : bool, optional
        Whether to set the time of the date object to midnight, default False

    date_only : bool, optional
        Whether to only return the datetime.date part of the datetime object

    Returns
    -------
    datetime.date or datetime.time
        Verified date object built from the string date_to_check
    """

    try:
        checked_date = datetime.datetime.strptime(date_to_check, expected_format)
        if set_to_midnight:  # optional replace time with 00:00:00, e.g. for file name purposes
            checked_date = checked_date.replace(hour=0, minute=0, second=0, microsecond=0)

        if date_only:
            checked_date = datetime.date(year=checked_date.year,
                                         month=checked_date.month,
                                         day=checked_date.day)
        return checked_date

    except ValueError:
        raise argparse.ArgumentTypeError(f"{date_to_check} does not conform to {expected_format} format")


def get_yaml_config(yaml_config_file):
    """ Does simple checks on the provided yaml configuration file and safe loads into dictionary

    Parameters
    ----------
    yaml_config_file : str
        Full path to YAML config file to check

    Returns
    -------
    dict
        The dictionary containing the loaded YAML options
    """

    # first check if it exists!
    yaml_config_file = check_yaml_exists(yaml_config_file)

    # now check we can parse it ok
    try:
        with open(yaml_config_file, 'r') as config:
            config_dict = yaml.safe_load(config)
            config_dict["yaml_config_file"] = yaml_config_file
            # add the file path into the dictionary in case the parent needs to access it

    except yaml.YAMLError as err:
        raise argparse.ArgumentTypeError(f"Configuration file contains invalid YAML:\n {str(err)}")

    return config_dict


def check_yaml_exists(yaml_file):
    """Check that the given yaml config file actually points towards a valid yaml config file

    Parameters
    ----------
    yaml_file : str
        The value of the full path to yaml config file

    Returns
    -------
    str
        The config path name, if all requirements are met
    """

    # check extension
    is_yaml = yaml_file.lower().endswith((".yml", ".yaml"))
    if not is_yaml:
        raise IOError(f"{yaml_file} is not a valid YAML file")

    # check if the file exists
    readable_file(yaml_file)

    return yaml_file


def change_directory_of_file_path(new_dir, file_path):
    """ Creates a path that has the same file name but different directory

    Parameters
    ----------
    new_dir : str
        New directory where to build file path to

    file_path : str
        Full path to a file

    Returns
    -------
    str
        A full path to a file location that has the same file name as the original, but a different directory
    """
    original_file_dir, original_file_base = get_file_directory_and_basename(file_path)
    new_file_path = os.path.join(new_dir, original_file_base)
    return new_file_path


def move_file(file_to_move, destination, create_destination=False):
    """ Moves file to given destination folder

    Parameters
    ----------
    file_to_move : str
        Path to file to move

    destination : str
        Path to the folder to move the file to

    create_destination : bool, optional
        Whether or not to create the destination folder, if it doesn't already exist

    Returns
    -------
    str
        path to the moved file
    """

    try:
        if create_destination:
            check_path_exists_and_create(destination)

        # check file and destination
        writeable_file(file_to_move)
        writeable_dir(destination)

    except IOError as err:
        raise IOError(f"Can't move file: {str(err)}")

    # do the move
    return shutil.move(file_to_move, destination)


def check_path_and_remove_tree(check_path):
    """ Checks that given path exists. If it does, removes that directory and all contents

    Parameters
    ----------
    check_path : str
        Path to check and create if not exists

    Returns
    -------
    bool
        Whether or not the path was deleted
    """
    try:
        shutil.rmtree(check_path)
        return True
    except (NameError, FileNotFoundError):
        # all good as the path doesn't exist
        return False


def check_path_exists_and_create(check_path):
    """ Checks that given path exists. If it doesn't, it creates the path.

    Parameters
    ----------
    check_path : str
        Path to check and create if not exists

    Returns
    -------
    None
    """
    try:
        os.makedirs(check_path)
    except FileExistsError:
        if os.path.isdir(check_path):
            # all good as the path already exists!
            pass
        else:
            raise IOError("Requested directory cannot be created as file exists with same name")


def check_file_and_delete(full_path):
    """ Checks that the file exists, and if so deletes it

    Parameters
    ----------
    full_path : str
        Full path to a file

    Returns
    -------
    bool
        Whether file was deleted or not
    """
    try:
        os.remove(full_path)
        return True
    except (NameError, FileNotFoundError):
        # all good as the file doesn't exist
        return False


def check_file_output_dir(full_path):
    """ Gets the directory from full file path, creates if doesn't exist

    Parameters
    ----------
    full_path : str
        Full path to a file

    Returns
    -------
    str
        Created directory where file can be saved
    """
    file_dir, _ = get_file_directory_and_basename(full_path)
    check_path_exists_and_create(file_dir)
    return file_dir


def check_directory(dir_string, access_type):
    """
    Determines if a string is a valid path to a directory with appropriate access

    Args:
        dir_string (str) : The path to check for validity
        access_type (int) : Any of os.W_OK, os.R_OK

    Raises:
        IOError : If path is not a valid directory
        IOError : If path is not readable directory
        ValueError : If access type is not one of (os.W_OK, os.R_OK)
    """
    if access_type not in (os.W_OK, os.R_OK):
        raise ValueError(f"Invalid access type {str(access_type)}")

    if not os.path.isdir(dir_string):
        raise IOError(f"'{dir_string}' is not a valid path")

    if not os.access(dir_string, access_type):
        check_type = ""
        if access_type == os.W_OK:
            check_type = "writeable"
        elif access_type == os.R_OK:
            check_type = "readable"
        raise IOError(f"dir: '{dir_string}' is not a {check_type} dir")

    return dir_string


def readable_dir(dir_string):
    """ Determines if a string is a valid path to a readable directory
    Args:
        dir_string (str) : The path to check for validity

    Raises:
        IOError : If path is not a valid directory or directory is not readable
    """
    return check_directory(dir_string, os.R_OK)


def writeable_dir(dir_string):
    """ Determines if a string is a valid path to a writeable directory
    Args:
        dir_string (str) : The path to check for validity

    Raises:
        IOError : If path is not a valid directory
        IOError : If path is not writeable
    """
    return check_directory(dir_string, os.W_OK)


def check_file(file_string, access_type):
    """
    Determines if a string is a valid path to a file with appropriate access

    Args:
        file_string (str) : The path to check for validity
        access_type (int) : Any of os.W_OK, os.R_OK

    Raises:
        IOError : If path is not a valid file
        IOError : If file is not of requested access type
        ValueError : If access type is not one of (os.W_OK, os.R_OK)
    """
    if access_type not in (os.W_OK, os.R_OK):
        raise ValueError(f"Invalid access type {str(access_type)}")

    if not os.path.isfile(file_string):
        raise IOError(f"'{file_string}' is not a valid path to file")

    # TODO Should probably check that effective UID is the same as UID before running os.access
    # TODO Assume euid == uid and gid ==e guid for now
    if not os.access(file_string, access_type):
        check_type = ""
        if access_type == os.W_OK:
            check_type = "writeable"
        elif access_type == os.R_OK:
            check_type = "readable"
        raise IOError(f"dir: '{file_string}' is not a {check_type} file")

    return file_string


def readable_file(file_string):
    """
    Determines if a string is a valid path to a readable file

    Args:
        file_string (str) : The path to check for validity

    Raises:
        IOError : If path is not a valid file or file is not readable
    """
    return check_file(file_string, os.R_OK)


def writeable_file(file_string):
    """
    Determines if a string is a valid path to a writeable file

    Args:
        file_string (str) : The path to check for validity

    Raises:
        IOError : If path is not a valid file
        IOError : If file is not writeable
    """
    return check_file(file_string, os.W_OK)


def append_to_file_name(file_name, to_append):
    """ Append to the end of a file's name and return the new name.
        TODO: this currently will not handle multi-extension filenames such as test.tif.aux.xml

    Parameters
    -----------
    file_name :    String
            Current file name including extension
    to_append :    String
            text to insert before extension
    """
    file_body, file_ext = os.path.splitext(file_name)
    new_file_name = f'{file_body}_{to_append}{file_ext}'
    return new_file_name


def get_utc_timestamp(time_format='%Y-%m-%d_%H-%M-%S'):
    """ Get the current UTC date and time as a formatted string. """
    utc_now = datetime.datetime.utcnow()
    timestamp = utc_now.strftime(time_format)
    return timestamp


def get_differences_between_sets(first_set, second_set):
    """
    Determines the differences between two sets of information

    Parameters
    -----------
    first_set :    Set
                            First Collection of information

    second_set :    Set
                            Second collection of information to compare with the first

    Returns
    ----------

    only_in_first :    Set
                                        Those records only in first comparison set

    only_in_second :    Set
                                        Those records only in second comparison set

    in_both :   Set
                                        Those records in both comparison sets

    """
    only_in_first = set(first_set).difference((set(second_set)))
    only_in_second = set(second_set).difference((set(first_set)))

    in_both = set(second_set).intersection((set(first_set)))

    return only_in_first, only_in_second, in_both


def check_if_path_begins_with_drive(path):
    """ True if starts like W:, False otherwise. Case insensitive
    Parameters
    ----------
        path (str) : The path of a folder or file to check for a drive letter
    Returns
    -------
        has_drive_letter : str
            Whether the path started with a drive letter or not
    """
    start = path[:2]
    has_drive_letter = start[0].isalpha()
    has_drive_colon = start[1] == ":"
    begins_with_drive = has_drive_letter and has_drive_colon
    return begins_with_drive


def get_unc_path_from_drive_path(drive_path):
    """
    Obtains the UNC Path from a drive path
    If not a network then returns the original drive path unmodified
    Parameters
    ----------
        drive_path (str) : The path of a folder or file with a drive letter
    Returns
    ---------
        str : The UNC path corresponding to the drive path given
    """
    if os.name == 'nt':
        drive, tail = os.path.splitdrive(drive_path)

        try:
            output = subprocess.check_output(("net", "use", drive), universal_newlines=True, stderr=subprocess.STDOUT)
            all_lines = output.splitlines()
            unc_section = drive  # default to drive if cannot find the unc section
            for line in all_lines:
                if line.startswith('Remote name'):
                    unc_section = line[line.index(r"\\"):]
            drive_path = unc_section + tail
        except subprocess.CalledProcessError:  # noqa: F841
            # No network drive so use the local drive
            pass
    else:
        raise NotImplementedError('This function is only available on Windows systems')

    return drive_path


def convert_to_utc(dt):
    """ Converts datetime to UTC second since 1970 01 01
    Warning: changes timezone info to UTC - may not be desired functionality!

    Parameters
    ----------
    dt : datetime.datetime
        Date object to convert

    Returns
    -------
    int
        UTC seconds since 1970-01-01
    """
    # get date in UTC, seconds since 1970-01-01
    # https://docs.python.org/3.3/library/datetime.html#datetime.datetime.timestamp
    utc = int(dt.replace(tzinfo=datetime.timezone.utc).timestamp())  # int to throw away microseconds
    return utc


def convert_from_utc(utc):
    """ Converts utc timestamp integer to datetime object

    Parameters
    ----------
    utc : int
        UTC timestamp seconds since 1970-01-01

    Returns
    -------
    datetime.datetime
        Datetime object of the UTC timestamp
    """
    if utc is not None:
        dt = datetime.datetime.fromtimestamp(utc)
    else:
        dt = None
    return dt


def get_date_n_days_ago(n):
    """ Returns datetime of date n days ago from today's date

    Parameters
    ----------
    n, int
        Number of days ago to get date for

    Returns
    -------
    datetime.datetime
    """
    date_midnight_today = datetime.datetime.combine(datetime.datetime.utcnow().date(), datetime.time.min)
    date_n_days_ago = date_midnight_today - datetime.timedelta(days=n)
    return date_n_days_ago


def get_date_n_hours_ago(n):
    """ Returns datetime n hours ago from now

    Parameters
    ----------
    n, int
        Number of hours ago to get datetime for

    Returns
    -------
    datetime.datetime
    """
    datetime_now = datetime.datetime.utcnow()
    datetime_n_hours_ago = datetime_now - datetime.timedelta(hours=n)
    return datetime_n_hours_ago


def get_file_directory_and_basename(full_path):
    path = pathlib.Path(full_path)
    file_dir = path.parent
    file_basename = path.name
    return file_dir, file_basename


def is_file_being_written(file_to_check, wait_time=2):
    """ Checks if given file is still being written to

    Parameters
    ----------
    file_to_check : str
        Full path to the file to check if still being written to

    wait_time : int, optional
        Number of seconds to wait to check file size

    Returns
    -------
    bool
        Whether or not the file is still being written to
    """
    file_size_1 = get_file_size(file_to_check)
    time.sleep(wait_time)
    file_size_2 = get_file_size(file_to_check)
    still_writing = file_size_2 != file_size_1
    return still_writing


class TwoSetEquality:
    def __init__(self, first_set, second_set):
        self.first_set = first_set
        self.second_set = second_set

        self.num_only_in_first = 0
        self.num_only_in_second = 0

        self.only_in_first = None
        self.only_in_second = None
        self.in_both = None

        self.sets_are_equal = self.are_sets_equal()

    def are_sets_equal(self):
        self.only_in_first, self.only_in_second, self.in_both = get_differences_between_sets(self.first_set,
                                                                                             self.second_set)
        self.num_only_in_first = len(self.only_in_first)
        self.num_only_in_second = len(self.only_in_second)

        if self.num_only_in_first > 0 or self.num_only_in_second > 0:
            return False
        else:
            return True


def convert_seconds(seconds):
    """ Converts seconds to a readable dd:HH:mm:ss string format

    Parameters
    ----------
    seconds : int or float
        Seconds to convert

    Returns
    -------
    str
        "dd days HH hrs mm mins ss secs"
    """
    day = seconds // (24 * 3600)
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return f"{int(day)}days {int(hour):02d}hrs {int(minutes):02d}mins {int(seconds):02d}secs"


def create_temp_dir(base_temp_dir, length=None):
    """ Creates a unique temporary directory using a uuid4 id
    """
    unique_id = uuid.uuid4().hex
    if length is not None:
        unique_id = unique_id[:length]
    temp_dir = os.path.join(base_temp_dir, unique_id)
    check_path_exists_and_create(temp_dir)
    return temp_dir


def generate_date_range(start_date, end_date=None, freq=None, str_date_format=None):
    """ Creates a list of dates between given start and end dates.  If no end date provided, returns a list with one
    item - the start date.

    Parameters
    ----------
    start_date : datetime.datetime or str
        The start date of the range.  If provided as string, must also provide str_date_format for datetime conversion

    end_date : datetime.datetime or str, optional
        The end date of the range.  If provided as string, must also provide str_date_format for datetime conversion

    freq : str, optional
        Time step used to create the date range. Uses Pandas time step aliases as listed here:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases

    str_date_format : str, optional
        If start and/oir end date provided as string, this is the datetime format string that will be used to convert to
        datetime object

    Returns
    -------
    list
        List of datetime objects
    """
    if isinstance(start_date, str):
        start_date = check_date(start_date, str_date_format)

    if end_date and isinstance(end_date, str):
        end_date = check_date(end_date, str_date_format)
        if end_date < start_date:
            raise ValueError(f"Given end date {end_date} is before start date {start_date}. Cannot create date range. ")

    if end_date:
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq).to_pydatetime().tolist()
    else:
        date_range = [start_date]

    return date_range


def unzip(zip_file, del_zip=True, unzip_dir=None):
    """ Extracts files contained within zip archive to specified directory.

    Parameters
    ----------
    zip_file : str
        Path to .zip archive file

    del_zip : bool, optional
        Whether or not to delete .zip file after extraction

    unzip_dir : str, optional
        Directory in which to extract files to.  If not provided, uses the directory where the .zip file is located.

    Returns
    -------
    str
        Path to directory where extracted files were saved
    """
    with zipfile.ZipFile(zip_file, "r") as zip_extractor:
        if unzip_dir is None:
            unzip_dir = check_file_output_dir(zip_file)
        zip_extractor.extractall(unzip_dir)

    if del_zip:
        # delete original zip
        check_file_and_delete(zip_file)

    return unzip_dir


def is_file_process_required(target_file, overwrite, log=True):
    """ Determines whether a process is required based on whether the target file that would be created by that
    process already exists, and taking into account whether an overwrite option has been specified
    """
    try:
        readable_file(target_file)
        if overwrite:
            if log:
                LOGGER.warning(f"Overwrite specified and existing file found at {target_file}. "
                               f"Existing file will be overwritten with new file")
            process_required = True
        else:
            if log:
                LOGGER.debug("Existing file found. Overwrite not specified.  Skipping process...")
            process_required = False

    except (IOError, ValueError):
        # file doesn't exist so need to process!
        process_required = True

    return process_required


def log_time_elapsed(log):
    """ Decorator function which writes to the given logger the amount of time a function (which is decorated with this
    decorator) has taken to run.

    Parameters
    ----------
    log : Logger
        The logging object to write to
    """
    def decorator(func):
        @wraps(func)  # Copies the decorated function name, docstring, arguments list, etc,
        # otherwise this stuff gets lost using 'normal' decorator syntax
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.utcnow()
            try:
                val = func(*args, **kwargs)
                return val
            except Exception:
                raise
            finally:
                end_time = datetime.datetime.utcnow()

                if func.__module__ != '__main__':
                    module = func.__module__
                    name = f"{module}.{func.__name__}"
                else:
                    name = func.__name__

                seconds_taken = (end_time - start_time).total_seconds()
                log.info(f"{name}: {convert_seconds(seconds_taken)} elapsed.")

        return wrapper
    return decorator


def log_function_wrap(log, char="-", length=40, start=True, end=True):
    """ Wraps the given function with a logging message of a line of characters.
    """
    def decorator(func):
        @wraps(func)  # Copies the decorated function name, docstring, arguments list, etc,
        # otherwise this stuff gets lost using 'normal' decorator syntax
        def wrapper(*args, **kwargs):
            if start:
                log.info(char * length)

            try:
                val = func(*args, **kwargs)
                return val
            except Exception:
                raise
            finally:
                if end:
                    log.info(char * length)

        return wrapper
    return decorator


def get_long_alphabet(num_letters=2):
    alphabet = list(string.ascii_uppercase)
    alpha_list = []
    for _num_letters in range(1, num_letters+1):
        alpha_list += ["".join(p) for _ in range(1, _num_letters + 1)
                       for p in itertools.product(alphabet, repeat=_num_letters)]
    return alpha_list


def retry(exceptions=Exception, retries=1, delay=0, max_delay=None, backoff=1, log=None):
    def decorator(func):
        @wraps(func)  # Copies the decorated function name, docstring, arguments list, etc,
        # otherwise this stuff gets lost using 'normal' decorator syntax
        def wrapper(*args, **kwargs):
            try:
                val = __retry(partial(func, *args, **kwargs), exceptions, retries, delay, max_delay, backoff, log=log)
                return val
            except Exception:
                raise
        return wrapper
    return decorator


def __retry(func, exceptions=Exception, retries=1, delay=0, max_delay=None, backoff=1, log=None):
    """ Executes given function, retrying if given exceptions are raised

    Parameters
    ----------
    func : functools.partial
        The function to execute, as a "partial" with args and kwargs

    exceptions : exception or tuple, optional
        Exceptions to catch and retry on.  Default is catch all Exception

    retries : int, optional
        Maximum number of retry attempts. Default is 1

    delay : int, optional
        Delay (in seconds) between attempts.  Default is 0 seconds, no delay

    max_delay : int, optional
        Maximum allowed delay to wait between attempts.  Only used if backoff is a positive integer, which will
        gradually increase the delay between attempts. Default is No Limit

    backoff : int, optional
        Multiplier applied to delay parameter on subsequent retry attempts.  Default is 1 (no backoff)

    log : Logger
        Optional logging object to write to
    """
    _retries, _delay = retries, delay
    try_num = 0
    while True:
        try:
            try_num += 1
            return func()
        except exceptions as err:
            if _retries == 0:
                if log is not None:
                    log.error(f"{err} | All retries attempted")
                raise err
            _retries -= 1

            if log is not None:
                if _delay == 0:
                    delay_text = ""
                else:
                    delay_text = "in {_delay} seconds... "
                log.warning(f"{err} | Retrying function {delay_text}| Attempt {try_num} of {retries}")

            time.sleep(_delay)
            _delay *= backoff

            if max_delay is not None:
                _delay = min(_delay, max_delay)


def check_if_historical(config, start_priority):
    """
    Checks if a job's output base dir based on its priority and updates the configuration accordingly. It runs
    historical control member only if priority=9, it runs all available historical members if priority=8, otherwise,
    it runs real-time glofas-v4 for production.

    If the job's priority is set to 8 or 9, this function will log the initiation of a historical event
    and update the 'output_base_dir' in the configuration to reflect a directory change from
    'forecasting' to 'historical'; otherwise 'output_base_dir' needs to be 'forecasting'.

    Parameters:
    - config (dict): Configuration dictionary containing settings such as output paths.
    - start_priority (int): task run order.

    Returns:
    - priority (int): updated task run order.
    """
    output_base_dir = config["output_config"]["output_base_dir"]

    if start_priority >= 100:
        priority, low_rp = split_digits(start_priority)
        # Update minimum_flow_rp to the one define from historical setup
        if "threshold_config" in config and "minimum_flow_rp" in config["threshold_config"]:
            config["threshold_config"]["minimum_flow_rp"] = low_rp
    else:
        priority = start_priority

    mode_map = {
        (2, 8, 9): ("historical", "historical reforecast"),
        (7,): ("reanalysis", "historical reanalysis"),
        (5, 6): ("historical", "historical non-glofas"),
    }

    # Determine target mode and message
    for keys, (target_mode, message) in mode_map.items():
        if priority in keys:
            LOGGER.info(f"Run {message} event priority={priority}...")
            break
    else:  # default / operational forecasting & monitoring
        target_mode = "forecasting"
        LOGGER.info(f"Run operational priority={priority}...")

    # --- special case: keep monitoring base folder for priority==0 ---
    if priority == 0 and ("monitoring" in output_base_dir  or "forecasting" in output_base_dir):
        LOGGER.info("Priority=0 with 'monitoring' or 'forecasting' base folder detected; leaving output_base_dir unchanged.")
        config["output_config"]["output_base_dir"] = output_base_dir
        return priority

    # Apply replacements
    for src in ("forecasting", "reanalysis", "monitoring", "historical"):
        if src in output_base_dir and src != target_mode:
            output_base_dir = output_base_dir.replace(src, target_mode)

    config["output_config"]["output_base_dir"] = output_base_dir

    return priority


def split_digits(combined_id):
    """
    Split a combined number back into its individual digits.

    Parameters:
    combined_id (int): The combined number to split.

    Returns:
    tuple: A tuple containing the two parts.
    """
    if not (100 <= combined_id <= 999):
        raise ValueError("Input must be a number in the range 100-999.")

    digit1 = combined_id // 100
    digit2 = combined_id % 100
    return digit1, digit2
