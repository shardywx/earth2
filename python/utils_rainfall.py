""" Functions common to the ETLs of rainfall data, for surface water Flood Foresight project
"""
import os
import itertools
import logging

import math
from datetime import timedelta, datetime

import numpy as np
import xarray
import pandas as pd

import etl_common
from etl_common import ETLError
import helpers_general, helpers_xarray, helpers_foresight

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

# netcdf variable names for output datasets
X_VAR = "projection_x_coordinate"
Y_VAR = "projection_y_coordinate"
TIME_VAR = "time"
TIME_BOUNDS_VAR = "time_bnds"
ACC_VAR = "accumulation_period"
RP_VAR = "return_period"
THRESHOLD_EXCEEDED_VAR = "ensembles_exceeding_threshold"
TOTAL_ENSEMBLES_VAR = "total_ensembles"

THRESHOLD_EXCEEDED_DTYPE = np.uint8  # may need to be uint16 if more than 255 ensembles.
                                     # this is okay for now. mogreps = 216 ensembles, ireps = 198


def get_latest_time(etl_config, run_date, end_run_date):
    now = datetime.now()

    try:
        top_data_path = os.path.join(etl_config.load_directory, etl_config.data_source)
        latest_datetime_object = helpers_foresight.get_src_latest_time(top_data_path)
        # check backwards 2 * etl_config.time_lag_forecasts hours if found latest src date
        latest_datetime_object = latest_datetime_object - timedelta(
            hours=((2 * etl_config.time_lag_forecasts)//etl_config.run_time_gap)*etl_config.run_time_gap)
    except ValueError:
        latest_datetime_object = now

    try:
        run_date = get_correct_time_str(run_date)
    except ValueError:
        run_date = now.strftime("%Y-%m-%d-00:00")

    run_date_object = datetime.strptime(run_date, '%Y-%m-%d-%H:00')
    if run_date_object > latest_datetime_object:
        run_date = latest_datetime_object.strftime('%Y-%m-%d-%H:00')

    if end_run_date is None:
        end_run_date = now.strftime("%Y-%m-%d-%H:00")
    else:
        end_run_date = get_correct_time_str(end_run_date)

    return run_date, end_run_date


def get_correct_time_str(target_date):
    input_date = target_date.split('-')
    if len(input_date) == 3:
        target_date = f"{input_date[0]}-{input_date[1]}-{input_date[2]}-00:00"
    elif len(input_date) == 4:
        if ':' in target_date:
            target_date = f"{input_date[0]}-{input_date[1]}-{input_date[2]}-{input_date[3]}"
        else:
            target_date = f"{input_date[0]}-{input_date[1]}-{input_date[2]}-{input_date[3]}:00"

    return target_date


def run_transfrom(etl_config, file_or_ds):
    """ Handles workflow to transform raw IREPS data into a dataset that describes the number of ensembles exceeding
     given thresholds for different return periods and accumulation periods

     Parameters
     ----------
     etl_config : IrepsConfig
        Custom class of configuration options

     file_or_ds : str or xarray data set
         str for file path to raw IREPS file that requires transformation or xarray dataset derived from raw data

     Returns
     -------
     xarray.Dataset
     """

    LOGGER.info(f'{"-" * 15} TRANSFORM {"-" * 15}')
    if etl_config.skip_transform:
        LOGGER.info(f"Skip transform specified. Transform and Load steps skipped.")
    else:
        if etl_config.skip_transform_1:
            LOGGER.info(
                f"Skip transform stage 1 specified.  Skipping stage 1 transformation and intermediate load.")
            ds_stage_1 = None
        else:
            LOGGER.info("TRANSFORM STEP 1/2")
            if file_or_ds is None:  # if extract step skipped, get expected filename for date
                file_or_ds = build_extracted_file_path(etl_config)

            ds_stage_1 = run_transform_stage_1(etl_config, file_or_ds)

            if ds_stage_1:
                LOGGER.info(f'{"-" * 15} INTERMEDIATE LOAD OF ENSEMBLE COUNT NETCDF {"-" * 15}')
                ds_stage_1 = run_load(etl_config, ds_stage_1, etl_stage_description="base_ensembles")

        if etl_config.skip_transform_2:
            LOGGER.info(f"Skip transform stage 2 specified.  Skipping stage 2 transformation and final load.")
        else:
            if ds_stage_1 is None:  # if first transform was skipped then we need to get the ds direct from file
                nc_file = etl_common.build_transformed_file_path(etl_config, etl_stage_description="base_ensembles")
                if os.path.exists(nc_file):
                    ds_stage_1 = xarray.load_dataset(nc_file)
                else:
                    LOGGER.info(f'Required file {nc_file} does not exist, skip stage 2 transform!')
                    return

            LOGGER.info(f'{"-" * 15} TRANSFORM STEP 2/2 {"-" * 15}')
            ds_stage_2 = run_transform_stage_2(etl_config, ds_stage_1)

            if ds_stage_2 is not None:
                LOGGER.info(f'{"-" * 15} FINAL LOAD OF ENSEMBLE COUNT NETCDF {"-" * 15}')
                run_load(etl_config, ds_stage_2, etl_stage_description="time_lagged_ensembles")


def run_transform_stage_1(etl_config, file_or_ds):
    """ Handles workflow to transform raw IREPS data into a dataset that describes the number of ensembles exceeding
    given thresholds for different return periods and accumulation periods

    Parameters
    ----------
    etl_config : IrepsConfig
       Custom class of configuration options

    file_or_ds : str or xarray data set
        str for file path to raw IREPS file that requires transformation or xarray dataset derived from raw data

    Returns
    -------
    xarray.Dataset
    """
    transformed_file = etl_common.build_transformed_file_path(etl_config, etl_stage_description="base_ensembles")
    transform_required = etl_common.is_etl_stage_required(etl_config, "transform", transformed_file)

    if transform_required:
        # get the raw data into an Xarray Dataset
        if is_xr_dataset(file_or_ds):
            LOGGER.info(f"Input {file_or_ds} is xarray dataset")
            ds = file_or_ds
        else:
            LOGGER.info(f"Transforming data from {file_or_ds}...")
            ds = xarray.open_dataset(file_or_ds)
        ds = ds.reset_coords()
        ds = ds.chunk(etl_config.chunk_dict)

        # reconfigure variable names to the common names
        ds = configure_variables(etl_config, ds)
        add_id_var = False
        add_max_ensembles = True
        if 'ireps' in etl_config.data_source:
            add_id_var = True
            add_max_ensembles = False
            # add time_bnds
            ds = add_time_bounds(ds)
            # convert accumulated rainfall since forecast start time, to accumulated rainfall over each forecast hour
            ds = convert_to_hourly_rain_accumulations(etl_config, ds)

        # remove the first time step as this will always be 0 everywhere - forecast start, so no accumulated rainfall
        n_time_steps = len(ds[TIME_VAR])
        ds = ds[{TIME_VAR: range(1, n_time_steps)}]

        LOGGER.info(f"Accumulating rainfall over specified accumulation periods...")
        ds = calculate_accumulations(etl_config, ds)

        LOGGER.info(f"Determining threshold exceedance counts...")
        ds = calculate_threshold_exceedance(etl_config, ds)

        LOGGER.info(f"Creating pseudo-ensembles with spatial fuzzying...")
        ds = spatial_fuzzying(ds)

        ds = ancillary_transforms(etl_config, ds, add_max_ensembles=add_max_ensembles, add_id_var=add_id_var)
        if 'mogreps' in etl_config.data_source:
            # Subtract 30 minutes from each time value
            ds[TIME_VAR] = ds[TIME_VAR] - pd.Timedelta(minutes=30)
    else:
        ds = None

    return ds


def is_xr_dataset(file_or_ds):
    try:
        if isinstance(file_or_ds, xarray.Dataset):
            return True
        elif isinstance(file_or_ds, str) and os.path.isfile(file_or_ds):
            return False
    except Exception as e:
        LOGGER.exception(f"Unable to determine dataset type: {file_or_ds}")
        raise e


def convert_to_hourly_rain_accumulations(etl_config, ds):
    """ Incoming IREPS rainfall values are in units of accumulated rain since the start of the forecast.  Convert to
    rainfall accumulation over each forecast hour

    e.g.
    hour               :   0   1   2   3   4   5
    rain (accumulation):   0   10  12  14  20  21
    rain               :   0   10  2   2   6   1
    """
    # shift to get values for one hour back
    rain_shifted = ds[etl_config.rain_var].shift({etl_config.time_var: 1}, fill_value=0)
    # and subtract current values with shifted values
    ds[etl_config.rain_var] = (ds[etl_config.rain_var] - rain_shifted)

    # divide by 1000 to convert mm to m.  (the threshold file is in units of meters)
    ds[etl_config.rain_var] = ds[etl_config.rain_var] / 1000.

    return ds


def run_transform_stage_2(etl_config, ds_stage_1):
    """ Handles workflow for second stage of transformation of raw IREPS data into a dataset that describes
    the number of ensembles exceeding given thresholds for different return periods and accumulation periods.
    This second stage adds the consideration of a time lagged ensemble count.
    """
    transformed_file = etl_common.build_transformed_file_path(etl_config, etl_stage_description="time_lagged_ensembles")
    transform_required = etl_common.is_etl_stage_required(etl_config, "transform", transformed_file)
    ds = None

    if transform_required:
        try:
            LOGGER.info(f"Running time lagged ensemble counts...")
            ds = time_lag_ensemble_count(etl_config, ds_stage_1)
        except ETLError as err:
            # don't have enough files to do time lagged ensemble count. this may be the case for the first few files
            # processed.
            LOGGER.warning(str(err))

    return ds


def run_load(etl_config, transformed_ds, etl_stage_description):
    """ Saves the current transformed dataset to file
    """
    if transformed_ds:
        if 'mogreps' in etl_config.data_source:
            # this removes unneeded coordinates from variables, e.g. forecast_ref_time from scalar variables
            transformed_ds = transformed_ds.reset_coords()

        netcdf_path = etl_common.build_transformed_file_path(etl_config, etl_stage_description=etl_stage_description)
        helpers_xarray.ds_to_netcdf(transformed_ds, netcdf_path, load=True, comp=True)
        LOGGER.info(f"Saving transformed data to {netcdf_path} with file size of {os.stat(netcdf_path).st_size}")
    else:
        LOGGER.warning("No dataset provided")

    return transformed_ds


def build_extracted_file_path(etl_config):
    """ Builds the path to the file name that the extracted dataset will be saved to

    Parameters
    ----------
    etl_config : irepsConfig
       Custom class of configuration options

    Returns
    -------
    str
        Full path to the extracted file
    """

    netcdf_output_dir = helpers_foresight.build_etl_dir(etl_config.extract_directory, etl_config.data_source,
                                                        "raw", file_date=etl_config.etl_datetime,
                                                        raw_date_format=etl_config.raw_date_format)
    helpers_general.check_path_exists_and_create(netcdf_output_dir)
    if 'ireps' in etl_config.data_source:
        netcdf_file = f"fc{etl_config.etl_datetime.strftime('%Y%m%d%H')}+000-" \
                      f"{etl_config.expected_lead_times-1:03d}_m000-m{etl_config.expected_ensembles-1:03d}.nc"
    elif 'mogreps' in etl_config.data_source:
        netcdf_file = f"{etl_config.etl_datetime.strftime('%Y%m%dT%H%MZ')}-PT0001H00M-" \
                      f"PT{etl_config.expected_lead_times:04d}H00M-rainfall-accumulation.nc"
    else:
        netcdf_file = ''
        LOGGER.warning("No matched data source provided")

    netcdf_full_path = os.path.join(netcdf_output_dir, netcdf_file)
    return netcdf_full_path


def configure_variables(etl_config, ds, additional_vars_to_keep=None):
    # rename variables as appropriate
    rename_dict = {etl_config.x_var: X_VAR, etl_config.y_var: Y_VAR, etl_config.time_var: TIME_VAR}
    # check the variables actually exist in dataset.  e.g. for threshold nc file, the time var won't exist
    for key in list(rename_dict.keys()):  # list() so we aren't changing key size during iteration.
        if key not in ds.variables:
            del rename_dict[key]
    if X_VAR not in list(ds.coords):
        ds = ds.rename(rename_dict)

    # remove unwanted variables
    current_vars = ds.variables
    final_vars = [etl_config.rain_var, etl_config.ensemble_var, etl_config.projection_var,
                  X_VAR, Y_VAR, TIME_VAR, TIME_BOUNDS_VAR]

    if additional_vars_to_keep:
        final_vars.extend(additional_vars_to_keep)

    diff = helpers_general.TwoSetEquality(current_vars, final_vars)
    ds = ds.drop_vars(diff.only_in_first)

    return ds


def add_time_bounds(ds):
    # If doesn't already exist, add time_bnds - a CF compliant variable that specifies the time 'period' over
    # which corresponding measurements are valid
    if not ds.get(TIME_BOUNDS_VAR):
        # create the time bounds data, assuming that the 'time' variable gives the 'upper' limit of the time bounds
        # and that data is hourly.
        # e.g. for a time value of 2020-01-01 12:00, corresponding bounds will be [2020-01-01 11:00, 2020-01-01 12:00]
        time_bnds_data = [[t - np.timedelta64(1, "h"), t] for t in ds[TIME_VAR].values]
        ds = ds.assign(variables={TIME_BOUNDS_VAR: ([TIME_VAR, "bnds"], time_bnds_data)})
        # set the attribute on the standard time variable so that it picks up the bounds e.g. in panoply
        ds["time"].attrs["bounds"] = TIME_BOUNDS_VAR

    return ds


def calculate_accumulations(etl_config, ds):
    """ Calculates accumulations of rainfall over a set number of accumulation periods.  Essentially sums up the
    rolling windows of rainfall
    """
    accumulations = []

    for hours in etl_config.threshold_acc_periods:
        accumulation = ds[etl_config.rain_var].rolling({etl_config.time_var: hours}).sum(min_count=hours, skipna=True)
        # for some reason, chunks get a bit messed up (e.g. for time length of 126, chunk would get set to 123)
        #   explicitly set to None to 'reset' the chunk
        accumulation = accumulation.chunk({etl_config.time_var: None})
        accumulations.append(accumulation)

    # concatenate the accumulation data together, creating a new dimension of 'accumulation_period'
    accumulation_arr = xarray.concat(accumulations, ACC_VAR)
    ds[etl_config.rain_var] = accumulation_arr

    ds = ds.assign_coords({ACC_VAR: etl_config.thresholds[ACC_VAR].values})  # add the dimension as a variable

    return ds


def calculate_threshold_exceedance(etl_config, ds):
    """ Determines how many of the dataset rainfall ensembles have exceeded pre-calculated return period thresholds.
    Thresholds have been calculated for specific return period and accumulation period combinations
    """
    # note, rainfall_accumulation could contain NAN (due to not enough time steps for accumulations early on in array)
    # the gte comparison will cause a runtime warning: invalid value encountered in greater_equal, but can just
    # ignore this
    thresholds_exceeded = ds[etl_config.rain_var] >= etl_config.thresholds[etl_config.threshold_rain_var]

    num_ensembles_exceeded = thresholds_exceeded.sum(dim=etl_config.ensemble_var)

    # explicitly stating dtype helps performance
    ds[etl_config.rain_var] = num_ensembles_exceeded
    ds[etl_config.rain_var] = ds[etl_config.rain_var].astype(THRESHOLD_EXCEEDED_DTYPE)

    ds = ds.rename({etl_config.rain_var: THRESHOLD_EXCEEDED_VAR})
    if etl_config.ensemble_var in ds.dims:
        ds = ds.drop(etl_config.ensemble_var)

    return ds


def spatial_fuzzying(ds, max_shift=1):
    """ For a given grid cell, sums up the value at that grid cell and all of it's neighbouring pixels.
    """
    # determines which surrounding pixels to include in the sum, based on the max_shift parameter.
    # max_shift=1 uses the surrounding 8 pixels
    shift_combos = itertools.product(range(-max_shift, max_shift+1), repeat=2)

    shifted_arrays = []
    for y_shift, x_shift in shift_combos:
        shifted_da = ds[THRESHOLD_EXCEEDED_VAR].shift({Y_VAR: y_shift, X_VAR: x_shift}, fill_value=0)
        arr = shifted_da.data.copy()
        arr[np.isnan(arr)] = 0
        shifted_arrays.append(arr)

    # this shift appears to change datatype from int to float64.  explicitly state the data type to maintain
    ds[THRESHOLD_EXCEEDED_VAR].data = sum(shifted_arrays)
    ds[THRESHOLD_EXCEEDED_VAR] = ds[THRESHOLD_EXCEEDED_VAR].astype(THRESHOLD_EXCEEDED_DTYPE)

    return ds


def time_lag_ensemble_count(etl_config, ds):
    """ Increase ensemble count in latest data by accumulating over previous forecast data for the corresponding
    forecast times

    Parameters
    ----------
    etl_config : ETLConfig
        Custom class of configuration options

    ds : xarray.Dataset
        The latest data to which we add the ensemble counts from previous forecasts

    Returns
    -------
    xarray.Dataset
        Time lagged ensemble count dataset
    """
    # get the intermediate transformed file paths for the lag times required, taking into account max acc. period
    max_acc_period = max(etl_config.threshold_acc_periods)
    n_files_required = math.ceil((max_acc_period - 1) / etl_config.run_time_gap) + (etl_config.time_lag_forecasts - 1)
    lag_files = {}
    num_missing_lags = 0
    for lag_time in range(1, n_files_required+1):
        target_tminus = lag_time * etl_config.run_time_gap
        lag_datetime = etl_config.etl_datetime - timedelta(hours=target_tminus)
        lag_file = etl_common.build_transformed_file_path(etl_config, etl_stage_description="base_ensembles",
                                                          override_datetime=lag_datetime)
        try:
            helpers_general.readable_file(lag_file)

        except (IOError, ValueError) as err:
            LOGGER.info(f"No file found for lag time T-{target_tminus}: {os.path.basename(lag_file)}")
            lag_file = None
            num_missing_lags += 1

        lag_files[target_tminus] = lag_file

    # now for each lag time file, we need to add the appropriate slices of the multi-dimensional "ensembles
    # exceeding thresholds" array to the 'base' array (that is the current "ensembles exceeding thresholds" array)

    # it is faster working directly on the numpy array at this point.  also, can't do index assignment on dask arrays.
    # we should have decreased data (both dimensions and storage - float to int) to be able to do this on np arrays now.
    base_arr = ds[THRESHOLD_EXCEEDED_VAR].data

    # needed to help define array slices
    n_time_steps = len(ds[TIME_VAR])

    for lag_idx, (target_tminus, lag_file) in enumerate(lag_files.items(), start=1):
        if lag_file is not None:
            # load the data from file and into numpy array
            LOGGER.debug(f"Processing file for lag T-{target_tminus}:  {lag_file}")
            lag_ds = xarray.open_dataset(lag_file)
            n_time_steps = min([len(lag_ds[TIME_VAR]), n_time_steps])  # make sure the lag file has teh same dimension
            lag_arr = lag_ds[THRESHOLD_EXCEEDED_VAR].data
            # for each accumulation period, we need to pick the appropriate slices from the arrays to sum together.
            LOGGER.info(f"Processing file for lag T-{target_tminus}:  {lag_file}")
            for acc_idx, acc_period in enumerate(etl_config.threshold_acc_periods):
                # determine indexes for the time dimension of the 'base' array (i.e. the latest data)
                # and the 'lag' array (i.e. the previous data from the lag file we are currently processing in the loop)

                # start index for the base array not required as will always start at the beginning of the dimension

                # an end index is required for the base array because the last few elements in the time dimension will not
                # have corresponding elements in the previous lag array due to the length of the forecast.  note that
                # this is also dependent on the desired number of files to lag over - for example lagging over 8 files
                # will mean that the time steps available in the base array will be constrained by the time steps available
                # in the lag array from 8 forecasts ago.
                idx_end_base = n_time_steps - (etl_config.run_time_gap * (etl_config.time_lag_forecasts - 1))

                # these are the indexes for the lag file.  the start index is whatever time step in the lag array
                # corresponds to the first time step in the base array - essentially the amount of time between forecasts
                idx_start = target_tminus
                # the end index is the corresponding time in the lag array to the end index of the base array
                idx_end = idx_end_base + target_tminus

                # these indexes are only applicable to those lag arrays which cover all the forecast times within the base
                # array (up to the end base index)
                if idx_end <= n_time_steps:
                    # arrays in format [acc, rp, time, y, x]
                    base_arr[acc_idx, :, :idx_end_base, :, :] += lag_arr[acc_idx, :, idx_start:idx_end, :, :]
                else:
                    pass

                # for those lag arrays that do not cover all the forecast times in the base array, we need to determine
                # different end indexes.  these will be for accumulation periods longer than 1 hour and lag files that
                # are greater than "n" forecasts ago, where n is the desired number of files to lag over.

                # For example, if we want to lag over 4 files, we will end up using lag files from more than 4 forecasts ago
                # due to the consideration of accumulation periods of e.g. 3 hours.  Consider a 3 hour accumulation for the
                # first time step of the base array (T+1) with a forecast model that runs every hour. In the base array
                # there is no data for this 3hr acc (as there haven't been 3 hours in the forecast to accumulate over).
                # Similarly in the base-1hr lag file, the corresponding time (T+2) again won't have data for the 3hr acc.
                # Therefore, to create a time-lag over 4 forecasts, we will need to use the 3hr acc data from base-2hr,
                # base-3hr, base-4hr and base-5hr lag files.

                if lag_idx >= etl_config.time_lag_forecasts:
                    idx_start = target_tminus
                    idx_end = (etl_config.time_lag_forecasts * etl_config.run_time_gap) + acc_period - 1

                    # this time, the end index for the 'base' array is dependent on how many time steps we are taking from
                    # the lag file
                    idx_end_base = idx_end - idx_start

                    if idx_start < idx_end:
                        # arrays in format [acc, rp, time, y, x]
                        base_arr[acc_idx, :, :idx_end_base, :, :] += lag_arr[acc_idx, :, idx_start:idx_end, :, :]
                    else:
                        pass

            del lag_ds, lag_arr  # explicit garbage collection of big stuff that is not required anymore

    # put the data back into the dataset, remove last few time steps that can't have time lag sums
    ds[THRESHOLD_EXCEEDED_VAR].data = base_arr
    max_time_steps = n_time_steps - (etl_config.run_time_gap * (etl_config.time_lag_forecasts - 1))
    ds = ds[{TIME_VAR: range(0, max_time_steps)}]

    # ensure datatype is maintained
    ds[THRESHOLD_EXCEEDED_VAR] = ds[THRESHOLD_EXCEEDED_VAR].astype(THRESHOLD_EXCEEDED_DTYPE)
    if 'mogreps' in etl_config.data_source:
        # Re-calculate and add the maximum number of ensembles that a count could be out of
        etl_config.total_ensembles = etl_config.total_base_ensembles * (etl_config.time_lag_forecasts - num_missing_lags)
    ds = ds.assign({TOTAL_ENSEMBLES_VAR: etl_config.total_ensembles})

    # make sure the grid mapping attribute persists
    ds[THRESHOLD_EXCEEDED_VAR].attrs["grid_mapping"] = etl_config.projection_var

    return ds


def ancillary_transforms(etl_config, ds, arrange_vars=True, add_id_var=True, add_max_ensembles=True, add_map_attr=True):
    """ Perform some 'bitty' tranformations on the data, processes which are isolated and don't fit in with any
    of the other main transformations
    """
    # reorder dimensions (ellipsis sticks the rest of the dimensions at the end in their original order)
    if arrange_vars:
        ds = ds.transpose(ACC_VAR, RP_VAR, TIME_VAR, Y_VAR, X_VAR, ...)

    # add the ID variable from the ID lookup netcdf
    # current (old) UK SW needs a grid ID look up table. The new MOGREPS grid is changed so the old look up table
    # is not going to work in this way but can be worked during the data transformation without the look up table,
    # so for the new MOGREPS data, set add_id_var to be False
    if add_id_var:
        ds = ds.merge(etl_config.id_lookup)

    # add an indication of the maximum number of ensembles that a count could be out of
    if add_max_ensembles:
        ds = ds.assign({TOTAL_ENSEMBLES_VAR: etl_config.total_base_ensembles})

    # add a grid mapping attribute, for compliance with Arc and CF conventions for projections
    # https://desktop.arcgis.com/en/arcmap/10.3/manage-data/netcdf/spatial-reference-for-netcdf-data.htm
    # http://cfconventions.org/cf-conventions/cf-conventions.html#attribute-appendix
    if add_map_attr:
        ds[THRESHOLD_EXCEEDED_VAR].attrs["grid_mapping"] = etl_config.projection_var

    return ds
