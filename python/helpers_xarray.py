""" Includes helper functions that configure, manipulate, analyse Xarray objects (data arrays, data sets).
Most likely to help with netCDF datasets.
"""
import numpy as np
import xarray
from dask.diagnostics import ProgressBar


def window(ds, var, dim, target_indices, fn="max", is_time_dim=True, add_time_bnds=True, time_delta_unit="h"):
    """ For reducing the size of a given dimension of a given variable within an Xarray dataset, you might want to
    'window' the data by, for example, taking the maximum of the variable's values over given dimension window/slices.

    e.g. For a variable with a time dimension containing hourly values over 5 days (dimension time length = 120)
    we might hold hourly data for the first 6 hours, then max over 3-hour periods for the next 12 hours,
    then max over 6-hour periods for the next 24 hours.. etc..

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset containing the variable to window

    var : str
        Name of the target variable to window

    dim : str
        Name of the target dimension contained within "var" to window over

    target_indices : iterable
        List of increasing index integers that are used to create the windows across the given "dim"
        For example, consider a dimension with length of 12.  The target_indices of [1,2,3,6,9,12] would create
        windows across the following slices: 0:1, 1:2, 2:3, 3:6, 6:9, 9:12

    fn : str
        Accumulation function to apply, currently implemented: max

    is_time_dim : bool
        Default is True to assume that the "dim" given is a time dimension.  set to False if not.

    add_time_bnds : bool
        If the dim is a time dimension, this specifies whether to reconfigure the associated time_bnds dimension so that
        the time bounds describe the windowed time slices correctly

    time_delta_unit : str, optional
        The numpy time delta unit which is used to generate time bounds, as described here:
        https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-and-timedelta-arithmetic
        default is "h" for hour

    Returns
    -------
    xarray.Dataset
    """
    implemented_fns = ["max"]
    if fn not in implemented_fns:
        raise NotImplementedError(f"Given function {fn} not implemented in window function")

    # this will hold the dataarrays for variable, with fn applied over given dimension windows
    dim_windows = []
    # this will hold the value of the 'new' dimension steps.  This will be the last step in a given window
    window_dim_values = []

    previous_window_step = 0
    for window_end_step in target_indices:
        # select the data for given time window
        ds_window = ds[var].isel({dim: slice(previous_window_step, window_end_step)})

        # apply given function over that window
        if fn == "max":
            window_result = ds_window.max(dim)
        else:
            raise NotImplementedError(f"Given function {fn} not implemented in window function")

        dim_windows.append(window_result)
        window_dim_values.append(ds_window[dim].data[-1])
        previous_window_step = window_end_step

    # join together all of the windows
    window_arr = xarray.concat(dim_windows, dim)

    # remove the window dimension from the original dataset, so that we can add our new data array that has a
    # different length and values of given dimension
    attrs = ds[dim].attrs  # save the previous attributes to add back on
    if is_time_dim:
        # time dimension's units and calendar may be held in the encoding dictionary
        encoding = {"units": ds[dim].encoding.get("units", None),
                    "calendar": ds[dim].encoding.get("calendar", None)}

    ds = ds.isel({dim: 0}).squeeze().drop(dim)  # remove the dimension

    # now add the new windowed max array into the dataset and set the values of the dimension accordingly
    ds[var] = window_arr
    ds = ds.assign_coords({dim: window_dim_values})  # add the dimension as a new variable with new dimension values

    # add the dimension variable attributes back on
    ds[dim].attrs = attrs
    if is_time_dim:
        ds[dim].encoding = encoding

        if add_time_bnds:
            # first, work out the length of each time slice
            slices = [0] + target_indices  # add 0 value at the start to be able to calculate first window length
            time_bnd_sizes = [slices[idx] - slices[idx - 1] for idx in range(1, len(slices))]
            # then calculate and add the time bounds data to bounds variable dimension
            ds = add_time_bounds(ds, dim, time_bnd_sizes, time_delta_unit=time_delta_unit)

    return ds


def add_time_bounds(ds, time_var, bnd_size, time_delta_unit="h"):
    """Add a time bounds variable/data, a CF compliant variable that specifies time 'period' over which
    corresponding measurements are valid.   Added as a bounds attribute of the time variable.

    Parameters
    ----------
    ds : xarray.Dataset
        Xarray dataset containing the time variable to add bounds to

    time_var : str
        Name of the time variable within ds

    bnd_size : int or array-like
        The size of the time bounds.  If integer, assumes same bound size for each time step.  If array-like (e.g. list
        or tuple), this must have the same length as the time variable in the ds and describe the length of each
        time bound at each individual time step

    time_delta_unit : str, optional
        The numpy time delta unit as described here:
        https://numpy.org/doc/stable/reference/arrays.datetime.html#datetime-and-timedelta-arithmetic
        default is "h" for hour

    Returns
    -------
    xarray.Dataset
        ds for time_bnds added
    """
    time_bnds_var = f"{time_var}_bnds"

    if isinstance(bnd_size, int):
        bnd_size = [bnd_size] * len(ds[time_var])

    if len(bnd_size) != len(ds[time_var]):
        raise ValueError(f"If bound_length given as iterable, needs to be same length as {time_var} variable "
                         f"within xarray dataset")

    # create the time bounds data, assuming that the 'time' variable gives the 'upper' limit of the time bounds
    time_bnds_data = []
    for upper_time, bound in zip(ds[time_var].values, bnd_size):
        lower_time = upper_time - np.timedelta64(bound, time_delta_unit)
        time_bnds_data.append([lower_time, upper_time])

    ds = ds.assign(variables={time_bnds_var: ([time_var, "bnds"], time_bnds_data)})
    # set the attribute on the standard time variable so that it picks up the bounds e.g. in panoply
    ds[time_var].attrs["bounds"] = time_bnds_var

    # time dimension's units and calendar held in the encoding dictionary. add to bounds.
    encoding = {"units": ds[time_var].encoding.get("units", None),
                "calendar": ds[time_var].encoding.get("calendar", None)}
    ds[time_bnds_var].encoding = encoding

    return ds


def filter_2D_cells_to_1D(ds, filter_variable, filter_values, x_var, y_var, stack_var="flat_coords"):
    """ For extracting specific grid cells from a dataset, based on the value of a dataset variable (that must
    have the spatial coordinates x and y).

    This creates a 1D spatial grid cell arrangement from the 2D x,y coordinates, and then filters the 1D grid cells
    using given target values.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to filter 2D spatial grid

    filter_variable : str
        Variable name of the dataset variable the filter will be based on

    filter_values : array-like
        Values of the variable that we are filtering on

    x_var : str
        Variable name of the X dimension

    y_var : str
        Variable name of the Y dimension

    stack_var : str
        Name of the new 1D "stack" variable that we convert the 2D grid to

    Returns
    -------
    xarray.Dataset
        Dataset with filtered, 1D spatial dimensions
    """

    # set all variables as coordinate variables, rather than data variables, because the filter below adds the stacked
    # dimension onto every data variable even if it doesn't need it.  e.g. a scalar data variable such as the projection
    # definition would be given a 1d dimension of length of the stacked dimension
    ds = ds.set_coords([v for v in ds.data_vars])

    # stack creates one-dimension from the x, y coordinates, so that we can actually reduce dataset when doing the
    # ds.where on rainfall cells.  otherwise, the data is still held in 2-dimensions and there are a lot of 'nan' cells
    ds = ds.stack({stack_var: (x_var, y_var)})

    # filter the dataset to only those cells we care about.
    # NOTE - this may not preserve original datatype, e.g. would convert int types to float64 due to the constraint
    #   of having NaNs with the "where" results (which do get dropped, but they are there in the interim stage).
    #   numpy has no NaN type for integers, so has to convert to float.
    orig_type = ds[filter_variable].dtype
    ds = ds.where(ds[filter_variable].isin(filter_values), drop=True)
    ds[filter_variable] = ds[filter_variable].astype(orig_type)  # convert back to original dtype

    # reset the one-dimensional spatial dimension to remove the 'sub indexes' of x and y
    ds = ds.reset_index(stack_var)

    # now we can reset the coordinate variables, which moves the non-coordinate variables back to being data variables
    ds = ds.reset_coords()

    return ds


def ds_to_netcdf(ds, nc_out, load=False, comp=True, comp_level=4, show_progress_bar=False, **kwargs):
    """ Wrapper function for saving xarray dataset to netcdf file.  Particularly useful when need to load dask
    arrays to memory and/or add compression to the netcdf file

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to save

    nc_out : str
        Full path of netcdf file to save as

    load : bool, optional
        Whether to run ds.load() to get dask arrays into memory.  Makes the save to file quicker, but requires enough
        cpu memory to hold the data

    comp : bool, optional
        Whether or not to add compression to the netcdf variables.  saves file size.

    comp_level : int, optional
        If adding compression, this is the level of compression to use.  Options are 1-9
        Higher numbers equals more compression

    show_progress_bar : bool, optional
        Whether or not to show the dask progress bar when loading ds to memory

    kwargs
        Any other of the keyword arguments used in the Xarray dataset.to_netcdf function

    Returns
    -------
    None
    """
    if load:
        # if chunked with dask, running 'load' first seems to speed things up on save
        if show_progress_bar:
            with ProgressBar():
                ds = ds.load()
        else:
            ds = ds.load()
    # save file. set compression settings to save a bit of disk space
    if comp:
        # this encoding specifies compression to apply to the netcdf file.
        #   see 'encoding' here: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_netcdf.html
        #   see 'writing encoded data' here: http://xarray.pydata.org/en/stable/io.html
        #   netCDF4: http://unidata.github.io/netcdf4-python/netCDF4/index.html#netCDF4.Dataset.createVariable
        comp_options = dict(zlib=True, complevel=comp_level)
        encoding = {var: comp_options for var in ds.data_vars}
    else:
        encoding = None

    ds.to_netcdf(nc_out, encoding=encoding, **kwargs)
