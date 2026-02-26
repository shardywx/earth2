import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
import seaborn as sns
import regionmask
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import math
from scipy.interpolate import PchipInterpolator   # shape-preserving monotonic interpolator

import matplotlib as mpl
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
try:
    # MetPy import (works for different MetPy versions)
    from metpy.plots import ctables
except Exception as exc:
    raise ImportError("MetPy is required to use MetPy color tables") from exc

import os
import re
import json
from pathlib import Path
from typing import Dict
sns.set_theme()


def load_config(config_path: str) -> Dict:
    """
    Load a configuration file containing relevant metadata such as file paths, event strings, etc.
    
    :param config_path: path to the config file 
    :type config_path: str
    :return: dictionary containing the configuration parameters and values from the JSON file
    :rtype: Dict
    """
    with open(config_path, 'r') as file: 
        config = json.load(file)
        return config 


def extract_seed(fname: str) -> int:
    """
    Exract seed number from the input filename (assumes format 'seedXX.nc' where XX is the seed number)
    
    :param fname: file name string containing the seed number (e.g. 'seed76.nc')
    :type fname: str
    :return: seed number extracted from the file name
    :rtype: integer
    """
    m = re.search(r'seed(\d+)', os.path.basename(fname))
    if not m:
        raise ValueError(f'No seed found in {fname}')
    return int(m.group(1))


def normalise_longitude(ds: xr.Dataset, 
                        lon_name='lon'):
    """ 
    Convert longitude from [0...360] to [-180...180] if necessary and sort the dataset by longitude.
    This ensures that the longitude coordinates are in a consistent format for plotting and analysis, 
    especially when working with global datasets that may use different longitude conventions. 
    """
    lon = ds[lon_name]
    if float(lon.max()) > 180:
        # map 0..360 -> -180..180 and re-sort
        ds = ds.assign_coords({lon_name: ((lon + 180) % 360) - 180}).sortby(lon_name)
    return ds


def select_subset_by_bbox(ds: xr.Dataset, 
                          bbox=[-11.0, 1.5, 50.25, 60.0]):
    """ 
    Select a subset of the dataset based on a bounding box (lon0, lon1, lat0, lat1). This is useful for focusing on a specific region (e.g. UK and Ireland) and speeding up processing and plotting.
    """
    # Determine latitude and longitude coord names
    if 'latitude' in ds.coords:
        lat_name = 'latitude'
    elif 'lat' in ds.coords:
        lat_name = 'lat'
    else:
        raise ValueError("No latitude coordinate found (expected 'lat' or 'latitude').")

    if 'longitude' in ds.coords:
        lon_name = 'longitude'
    elif 'lon' in ds.coords:
        lon_name = 'lon'
    else:
        raise ValueError("No longitude coordinate found (expected 'lon' or 'longitude').")

    # Ensure latitudes are ordered from south -> north (ascending)
    ds = ds.sortby(lat_name)

    # latitude and longitude bounds for UK and Ireland region (lon0, lon1, lat0, lat1)
    lon0, lon1, lat0, lat1 = bbox
    ds = ds.sel({lat_name:slice(lat0,lat1), lon_name:slice(lon0,lon1)})

    return ds, lat_name, lon_name


def calculate_rolling_total(ds: xr.Dataset | xr.DataArray, 
                            rolling_window: int,
                            min_periods: int,
                            lead_dim: str = 'valid_time') -> xr.DataArray:
    """ 
    Calculate a rolling sum over a given time dimension for a dataset.
    """

    if rolling_window is not None:
        if lead_dim is None:
            raise ValueError("Cannot compute rolling window without a lead/time dimension.")
        # rolling produces same-length lead dimension with NaNs in first (rolling_window-1) positions
        da_roll = ds.rolling({lead_dim: rolling_window}, min_periods=min_periods).sum()
    else:
        # assume da_s already is 24h totals with a lead_dim
        da_roll = ds
    return da_roll


def get_metpy_colourmap(name='precipitation'):
    """
    Return a matplotlib Colormap object for a MetPy colortable name.
    Handles several MetPy API return formats.
    """
    cmap_src = None
    # try possible APIs
    if hasattr(ctables, 'get_colortable'):
        cmap_src = ctables.get_colortable(name)        # newer API sometimes returns colormap or list
    elif isinstance(ctables, dict) and name in ctables:
        cmap_src = ctables[name]                       # older API returns list or tuple
    else:
        # try other attributes (some versions expose colortables dict)
        try:
            cmap_src = ctables.colortables[name]
        except Exception:
            raise KeyError(f"Could not find MetPy colortable '{name}' in this MetPy version")

    # If MetPy already returned a Colormap, use it
    if isinstance(cmap_src, mpl.colors.Colormap):
        return cmap_src

    # If it returned a tuple like (colors, bounds), extract colors
    if isinstance(cmap_src, tuple) and len(cmap_src) >= 1:
        colors = cmap_src[0]
    else:
        colors = cmap_src

    # Convert colors to numpy array
    colors = np.asarray(colors)

    # If colors are Nx3 or Nx4 integer 0..255, scale to 0..1
    if np.issubdtype(colors.dtype, np.integer):
        if colors.max() > 1:
            colors = colors.astype(float) / 255.0

    # If colors are Nx3 or Nx4 floats already 0..1, accept
    # If colors are hex strings, ListedColormap can handle them directly

    # Finally create a ListedColormap
    try:
        cmap = ListedColormap(colors, name=name)
    except Exception:
        # Fallback: try converting each color to an mpl color string
        cmap = ListedColormap([mpl.colors.to_rgba(c) for c in colors], name=name)

    # set NaN color, under/over if you like
    cmap.set_bad('lightgray')   # NaNs
    cmap.set_under('white')     # < vmin
    cmap.set_over('black')      # > vmax

    return cmap


def land_sea_mask(da: xr.DataArray | xr.Dataset) -> xr.DataArray:
    """ 
    Develop a land-sea mask for SFNO forecasts using regionmask and Natural Earth land polygons. 
    This will allow us to exclude ocean points from our analysis, which is important for precipitation forecasts where we are primarily interested in impacts on land.
    """

    # Determine latitude and longitude coord names
    if 'latitude' in da.coords:
        lat_name = 'latitude'
    elif 'lat' in da.coords:
        lat_name = 'lat'
    else:
        raise ValueError("No latitude coordinate found (expected 'lat' or 'latitude').")

    if 'longitude' in da.coords:
        lon_name = 'longitude'
    elif 'lon' in da.coords:
        lon_name = 'lon'
    else:
        raise ValueError("No longitude coordinate found (expected 'lon' or 'longitude').")

    # create an xarray 2D grid for regionmask
    lon = da[lon_name].values
    lat = da[lat_name].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    # get Natural Earth "land" polygon(s) at your desired resolution
    # regionmask has several options; this uses natural_earth with "land_50"
    land = regionmask.defined_regions.natural_earth_v5_1_2.land_50

    # make a mask: result dims (lat, lon) with 1=land, 0=sea, -1 outside region if masked
    mask = land.mask(lon2d, lat2d)   # returns numpy array of region integer indices or -1 where not in region

    # boolean land mask
    land_bool = (mask == 0)  # if 'land' region index is 0; inspect mask.unique() if uncertain

    # convert to xarray DataArray
    land_mask_da = xr.DataArray(land_bool.astype(bool),
                                coords={lat_name: lat, lon_name: lon},
                                dims=(lat_name, lon_name))

    # apply to precip DataArray da (with dims including lat/lon)
    da_land = da.where(land_mask_da) # previously applied to 'da_pr' i.e. only 4 ensembles 
    return da_land


def preprocess_input_nc_files(files: list,
                              rolling_window: int,
                              min_periods: int,
                              event_str: str,
                              bbox: list,
                              lead_dim='lead_time'):
    """
    Read in input files using xarray and modify coordinates and dimensions before plotting  
    """
    
    # TODO: build in functionality to deal with corrupted input files
    if len(files) > 1:
        datasets = []
        seeds = []
        for f in files:
            seed = extract_seed(f)
            ds = xr.open_dataset(f).squeeze('time')
            ds, _, lon_name = select_subset_by_bbox(ds,bbox=bbox)  # subset to region of interest and get lat/lon coord names
            ds = land_sea_mask(ds)  # mask out ocean points
            datasets.append(ds)
            seeds.append(seed)
        ds = xr.concat(datasets, dim=pd.Index(seeds, name='seed'))
        ds = normalise_longitude(ds, lon_name=lon_name)
    else:
        ds = xr.open_dataset(files[0])
        ds, _, lon_name = select_subset_by_bbox(ds,bbox=bbox)  # subset to region of interest and get lat/lon coord names
        ds = land_sea_mask(ds)  # mask out ocean points
        ds = normalise_longitude(ds, lon_name=lon_name)

    stack_dims = [d for d in ('seed', 'ensemble') if d in ds.dims]

    # check for singleton time dimensions and squeeze 
    # singleton_dims = [dim for dim, size in ds.sizes.items() if size == 1] 
    singleton_dims = [] 
    for dim, size in ds.sizes.items():
        if size == 1:
            singleton_dims.append(dim)
    if 'time' in singleton_dims:
        ds = ds.squeeze(dim='time')

    if len(stack_dims) > 1:
        # stack seed & ensemble into one member axis
        da_mslp = ds['msl'].stack(member=tuple(stack_dims))
        da_pr   = ds['tp06'].stack(member=tuple(stack_dims))
    elif len(stack_dims) == 1:
        # rename the single sampling dim to 'member' for consistency
        da_mslp = ds['msl'].rename({stack_dims[0]: 'member'})
        da_pr   = ds['tp06'].rename({stack_dims[0]: 'member'})
    else:
        # no sampling dims found — still ok, create a member dim of length 1 if you prefer
        da_mslp = ds['msl'].expand_dims(member=[0])   # optional
        da_pr   = ds['tp06'].expand_dims(member=[0])

    # convert from Pa to hPa
    da_mslp = da_mslp / 100.0
    # change units of 6-h acc precip from 'm' to 'mm'
    da_pr = da_pr * 1000.0

    # calculate rolling total (if applicable)
    da_pr_roll = calculate_rolling_total(da_pr,
                                         rolling_window=rolling_window,
                                         min_periods=min_periods,
                                         lead_dim=lead_dim)

    # remove the last 5 ensemble forecasts (do not contain real data)
    if event_str == 'Desmond_seed76':
        da_pr_roll = da_pr_roll.sel(member=slice(0,24))

    return da_pr_roll, da_mslp


def preprocess_era5_rainfall(ds: xr.Dataset,
                             rolling_window: int,
                             min_periods: int,
                             event_end_date = '2015-12-06T00',
                             bbox = [-12.0, 2.5, 48.0, 60.0],
                             lead_dim = 'valid_time') -> xr.DataArray:
    """ 
    Pre-process ERA5 rainfall data to be in the correct format for calculating threshold exceedance. This includes:
        - converting units from m to mm
    """

    # check for singleton time dimensions and squeeze 
    # singleton_dims = [dim for dim, size in ds.sizes.items() if size == 1] 
    singleton_dims = [] 
    for dim, size in ds.sizes.items():
        if size == 1:
            singleton_dims.append(dim)
    if 'time' in singleton_dims:
        ds = ds.squeeze(dim='time')

    # select subset of data for UK and Ireland region to speed up processing and plotting
    ds, _, lon_name = select_subset_by_bbox(ds, bbox=bbox)

    # normalise longitude if required (e.g. if dataset uses 0..360 but we want -180..180 for plotting)
    ds = normalise_longitude(ds, lon_name=lon_name)

    # change units of precip from 'm' to 'mm'
    # TODO: generalise for different variable names and units, e.g. by passing in the variable name and expected input and output units as parameters
    da_pr = ds['tp'] * 1000.0

    # calculate rolling totals if required (e.g. to get 24-h totals from hourly data)
    da_pr_roll = calculate_rolling_total(da_pr, 
                                         rolling_window=rolling_window, 
                                         min_periods=min_periods, 
                                         lead_dim=lead_dim)

    # select date and accumulation period (e.g. 24-h, 48-h)
    # default is for the period ending at 'event_end_date'
    da_pr_roll = da_pr_roll.sel(valid_time=(event_end_date))

    # assign relevant metadata as attributes for later use
    da_pr_roll.attrs['accumulation_period'] = rolling_window
    da_pr_roll.attrs['event_end_date'] = event_end_date

    return da_pr_roll


def format_accumulation_str(acc_period):
    """
    Format the accumulation period (e.g. 24) into a string for the title (e.g. '24-h').
    """
    if acc_period is None:
        return None
    # handle numeric values (int, float, numpy numeric types)
    if isinstance(acc_period, (int, float, np.integer, np.floating)):
        return f"{int(acc_period)}-h"
    # convert strings like '48h' or '48 h' to '48-h'
    s = str(acc_period).strip()
    # if s is numeric (e.g. '48'), append '-h'
    if s.isdigit():
        return f"{int(s)}-h"
    return s


def quantile_map(sfno_vals,
                 era5_vals, 
                 n_quantiles=11, 
                 method='pchip'):
    """
    Quantile mapping of sfno_vals to era5_vals.
    Both inputs are 1D arrays (pooled).
    Returns mapped array (same shape as sfno_vals) and the mapping function.

    - Use a `SciPy` interpolation method (`https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html`) that preserves the distribution shape
    - The quantile mapping functions that I originally thought I needed do not seem viable, because they require a time dimension (e.g. adjusting a climate model time series towards ERA5)
    - `https://python-cmethods.readthedocs.io/en/stable/methods.html`
    - `https://python-cmethods.readthedocs.io/en/stable/introduction.html`
    - `https://xclim.readthedocs.io/en/stable/notebooks/extendxclim.html`
    """
    # drop NaNs
    sfno = sfno_vals[np.isfinite(sfno_vals)]
    era5 = era5_vals[np.isfinite(era5_vals)]

    # sort values monotonically
    sfno = np.sort(sfno)
    era5 = np.sort(era5)

    qs = np.linspace(0, 1, n_quantiles)
    sfno_q = np.quantile(sfno, qs)
    era5_q = np.quantile(era5, qs)

    # ensure monotonic increasing (quantiles should be monotonic by construction)
    # build mapping function: from sf value -> era5 value
    # Use PCHIP for monotonic interpolation with sensible extrapolation
    if method == 'pchip':
        mapper = PchipInterpolator(sfno_q, era5_q, extrapolate=True)
    else:
        # numpy.interp fallback (linear; will clip at ends)
        def mapper(x):
            return np.interp(x, sfno_q, era5_q, left=era5_q[0], right=era5_q[-1])

    # apply mapping (preserve NaNs)
    mapped = np.full_like(sfno_vals, np.nan, dtype=float)
    mask = np.isfinite(sfno_vals)
    mapped[mask] = mapper(sfno_vals[mask])
    return mapped, mapper


def sanitize_and_write(ds_out,
                       outpath,
                       varname=None,
                       compress=True,
                       complvl=4):
    """
    Clean xarray.Dataset coords and attrs to make them safe for to_netcdf,
    then write to outpath with optional compression.
    Improvements:
      - convert object/tuple coords (including MultiIndex) to strings,
      - convert 0-d numpy arrays to Python scalars,
      - convert array-like attrs to strings or scalars,
      - cast data variables to float32,
      - preserve dimension coords (convert them) instead of trying to drop them.
    """
    import pandas as _pd

    ds = ds_out.copy()

    # --- clean coordinates ---
    for cname in list(ds.coords):
        coord = ds.coords[cname]
        val = coord.values

        # If coordinate is a pandas MultiIndex (common after stacking), convert to string labels
        try:
            if isinstance(coord.to_index(), _pd.MultiIndex):
                labels = ["/".join(map(str, t)) for t in coord.to_index()]
                ds = ds.assign_coords({cname: (coord.dims, np.array(labels))})
                continue
        except Exception:
            # not a MultiIndex; continue to other checks
            pass

        # object dtype arrays (likely arrays of tuples or mixed types)
        if isinstance(val, np.ndarray) and val.dtype == object:
            if val.size == 1:
                # single element object array: convert to python scalar string
                try:
                    scalar = val.item()
                    if isinstance(scalar, (tuple, list)):
                        label = str(tuple(int(x) if hasattr(x, '__int__') else x for x in scalar))
                    else:
                        label = str(scalar)
                    ds = ds.assign_coords({cname: label})
                except Exception:
                    # fallback: drop coordinate variable (leave dimension)
                    try:
                        ds = ds.drop_vars(cname, errors='ignore')
                    except Exception:
                        pass
            else:
                # convert each element to a safe string representation
                try:
                    labels = np.array([str(x) for x in val])
                    ds = ds.assign_coords({cname: (coord.dims, labels)})
                except Exception:
                    # last resort: drop the problematic coordinate variable
                    ds = ds.drop_vars(cname, errors='ignore')
            continue

        # numeric 0-d arrays -> convert to python scalar
        if isinstance(val, np.ndarray) and val.size == 1:
            try:
                scalar = val.item()
                ds = ds.assign_coords({cname: scalar})
            except Exception:
                ds = ds.assign_coords({cname: str(val.ravel()[0])})
            continue

        # numpy scalar (np.generic)
        if isinstance(val, (np.generic,)) and np.isscalar(val):
            try:
                ds = ds.assign_coords({cname: val.item()})
            except Exception:
                ds = ds.assign_coords({cname: str(val)})

    # --- special handling for a problematic "member" coord ---
    # If 'member' is present and contains tuples or object types, convert to safe string labels.
    if 'member' in ds.coords:
        member_coord = ds.coords['member']
        try:
            # If it's MultiIndex, convert to strings
            if hasattr(member_coord, 'to_index') and isinstance(member_coord.to_index(), _pd.MultiIndex):
                labels = ["/".join(map(str, t)) for t in member_coord.to_index()]
                ds = ds.assign_coords(member=(member_coord.dims, np.array(labels)))
            else:
                # If dtype is object or elements are tuples, convert each element to string
                mc_vals = member_coord.values
                if isinstance(mc_vals, np.ndarray) and mc_vals.dtype == object:
                    labels = np.array([str(x) for x in mc_vals])
                    ds = ds.assign_coords(member=(member_coord.dims, labels))
                # if it's 0-d or scalar, coerce to python scalar
                elif isinstance(mc_vals, np.ndarray) and mc_vals.size == 1:
                    ds = ds.assign_coords(member=mc_vals.item())
                # else leave as-is (it's likely numeric and safe)
        except Exception:
            # fallback: try to drop the member variable (not the dimension)
            try:
                ds = ds.drop_vars('member', errors='ignore')
            except Exception:
                pass

    # --- clean attributes ---
    for k, v in list(ds.attrs.items()):
        if isinstance(v, np.ndarray):
            if v.size == 1:
                try:
                    ds.attrs[k] = v.item()
                except Exception:
                    ds.attrs[k] = str(v.ravel().tolist())
            else:
                ds.attrs[k] = str(v.tolist())
        elif isinstance(v, (list, tuple, dict)):
            ds.attrs[k] = str(v)
        else:
            # leave scalar/string attrs untouched
            pass

    # --- ensure data variables are numeric and cast to float32 where sensible ---
    for var in list(ds.data_vars):
        try:
            ds[var] = ds[var].astype('float32')
        except Exception:
            # skip non-numeric variables
            continue

    # --- build encoding for to_netcdf ---
    if varname is None:
        data_vars = list(ds.data_vars)
    else:
        data_vars = [varname] if varname in ds.data_vars else list(ds.data_vars)

    encoding = {}
    for v in data_vars:
        if compress:
            encoding[v] = {'zlib': True, 'complevel': complvl, 'dtype': 'float32'}
        else:
            encoding[v] = {'dtype': 'float32'}

    # --- finally write ---
    ds.to_netcdf(outpath, mode='w', format='NETCDF4', encoding=encoding)
    return outpath


def format_accumulation_str(acc_period):
    """
    Format the accumulation period (e.g. 24) into a string for the title (e.g. '24-h').
    """
    if acc_period is None:
        return None
    # handle numeric values (int, float, numpy numeric types)
    if isinstance(acc_period, (int, float, np.integer, np.floating)):
        return f"{int(acc_period)}-h"
    # convert strings like '48h' or '48 h' to '48-h'
    s = str(acc_period).strip()
    # if s is numeric (e.g. '48'), append '-h'
    if s.isdigit():
        return f"{int(s)}-h"
    return s


def formatted_uk_and_ireland_plot(ds: xr.DataArray | xr.Dataset,
                                  bbox=[-12, 2.5, 48.0, 60.0],
                                  cmap=get_metpy_colourmap('precipitation')):  
    """ 
    Create a formatted contour plot for the UK and Ireland region, using Cartopy for mapping and MetPy color tables for styling.
    TODO: add an argument to choose type of plot (contourf, pcolormesh) independent of the metric we're plotting 
    """
    crs = ccrs.PlateCarree()
    cmap = cmap if cmap is not None else get_metpy_colourmap('precipitation')

    # ensure cmap is a Colormap instance (or get base cmap by name)
    if isinstance(cmap, str):
        base_cmap = plt.get_cmap(cmap)
    else:
        base_cmap = cmap

    # accept dataset with variable of interest, or DataArray directly
    if isinstance(ds, xr.Dataset):
        if len(ds.data_vars) == 1:
            var_name = list(ds.data_vars)[0]
            da = ds[var_name]
        else:
            raise ValueError("Dataset contains multiple data variables. Please provide a DataArray or a Dataset with only one data variable.")
    else:
        da = ds

    # format the title for the plot based on the variable name and coordinates (e.g. return level with return period coordinate, or accumulation period)
    var_name = None

    # determine whether we are plotting return level data 
    if 'rl' in da.coords:
        return_level = int(da['rl'].values)
        var_name = f'{return_level} year return levels'

    # or if we are plotting mapped return period data
    elif da.name == 'mapped_return_period':
        var_name = 'Max RP exceeded (years)'

    # or finally, accumulated precipitation (24-h, 48-h, etc.)
    else:
        acc_period = None
        # try to detect accumulation period from attributes or coordinates
        if 'accumulation_period' in da.attrs:
            acc_period = da.attrs['accumulation_period']
        elif 'accumulation_period' in da.coords:
            acc_period = da['accumulation_period'].values
        acc_str = format_accumulation_str(acc_period)
        if acc_str:
            var_name = f'{acc_str} accumulated precip (up to {da.attrs.get("event_end_date", "event end")})'
        else:
            var_name = da.name if da.name else '24-h accumulated precip'

    # select subset of data for UK and Ireland region to speed up processing and plotting
    ds, lat_name, lon_name = select_subset_by_bbox(ds, bbox=bbox)

    # If there are extra dimensions (e.g. time, ensemble), select the first index along them to get 2D
    other_dims = [d for d in da.dims if d not in (lat_name, lon_name)]
    for d in other_dims:
        da = da.isel({d: 0})

    # extract latitude/longitude arrays and create 2D lat/lon arrays for plotting
    lon = da[lon_name].values
    lat = da[lat_name].values
    Lon, Lat = np.meshgrid(lon, lat)

    # common function to add coastal features
    def decorate(ax):
        #ax.set_extent(extent, crs=crs)
        ax.coastlines(resolution='10m', linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.LAND.with_scale('10m'), facecolor='lightgray', zorder=0)


    # create a 'panel' plot with a single panel
    fig, axes = plt.subplots(figsize=(9, 6),
                             subplot_kw={'projection': crs})

    # Mean sea level pressure (pmsl)
    ax = axes
    decorate(ax)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.6, linestyle='--')
    
    # remove lat-lon labels on top and right axes 
    try:
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
    except AttributeError:
        gl.xlabels_top = False
        gl.xlabels_bottom = True
        gl.ylabels_left = False
        gl.ylabels_right = True

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # make the lat/lon labels smaller and lighter
    gl.xlabel_style = {'size': 8, 'color': 'darkslategray'}
    gl.ylabel_style = {'size': 8, 'color': 'darkslategray'}

    # set the levels for the contour plot based on the min and max values in the data, with a fixed interval (e.g. 1 mm)
    values = da.values.copy()

    # ensure cmap is a Colormap instance (handles strings)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    # If get_metpy_colourmap might return a name, also handle that:
    if not hasattr(cmap, 'N'):
        cmap = plt.get_cmap(str(cmap))

    # --- plotting: contourf for precipitation, pcolormesh otherwise ---
    mappable = None
    rp_ticks = None  # will be set for discrete RP maps

    # TODO: tidy this function to allow the user to decide what type of plot (contourf, pcolormesh) to choose, irrespective of the metric plotted
    if 'precip' in var_name:
        # Precipitation: define 1 mm levels and use contourf so level boundaries are clear
        pr_min = np.nanmin(values)
        pr_max = np.nanmax(values)
        interval = 1.0
        lev_min = np.floor(pr_min / interval)
        lev_max = np.ceil(pr_max / interval)
        pr_levels = np.arange(lev_min * interval, lev_max * interval + 1e-6, interval)

        #mappable = ax.contourf(Lon, Lat, values, levels=pr_levels, cmap=cmap, alpha=0.8, transform=crs)
        mappable = ax.pcolormesh(Lon, Lat, values, cmap=cmap, shading='auto', transform=crs)

    else:
        # Non-precip: use pcolormesh
        # Detect whether this is a mapped (integer) return-period map
        is_mapped_rp = (da.name == 'mapped_return_period') or (var_name == 'Max RP exceeded (years)')

        if is_mapped_rp:

            # canonical RP list (update if your RP set differs)
            rp_levels_full = np.array([2, 5, 10, 20, 30, 50, 75, 100, 200, 500, 1000, 1500])

            # flatten and get non-nan unique values in the plotted result
            flat_vals = values.ravel()
            non_nan = flat_vals[~np.isnan(flat_vals)]

            if non_nan.size == 0:
                # no valid data; fallback to continuous pcolormesh
                mappable = ax.pcolormesh(Lon, Lat, values, cmap=cmap, shading='auto', transform=crs)
                rp_ticks = None
            else:
                unique_rps_present = np.unique(non_nan.astype(int))
                max_present = unique_rps_present.max()

                # select canonical RP values up to and including the maximum present
                rp_vals = rp_levels_full[rp_levels_full <= max_present]
                if rp_vals.size == 0:
                    # nothing from canonical list less-equal max_present? fall back to unique present
                    rp_vals = np.sort(unique_rps_present)

                # ensure ascending
                rp_vals = np.sort(rp_vals).astype(float)
                n_bins = len(rp_vals)

                # build discrete colormap (one color per RP)
                base_cmap = cmap if hasattr(cmap, 'N') else plt.get_cmap(str(cmap))
                colors = base_cmap(np.linspace(0, 1, n_bins))
                discrete_cmap = ListedColormap(colors)

                # build boundaries half-way between values (handles irregular spacing)
                if n_bins == 1:
                    # single-category case: choose a sensible half-width using the canonical list spacing
                    idx0 = np.where(rp_levels_full == rp_vals[0])[0]
                    if idx0.size and idx0[0] < (len(rp_levels_full) - 1):
                        half_width = 0.5 * (rp_levels_full[idx0[0] + 1] - rp_levels_full[idx0[0]])
                    elif idx0.size and idx0[0] > 0:
                        half_width = 0.5 * (rp_levels_full[idx0[0]] - rp_levels_full[idx0[0] - 1])
                    else:
                        half_width = 0.5  # fallback
                    bounds = np.array([rp_vals[0] - half_width, rp_vals[0] + half_width])
                else:
                    midpoints = 0.5 * (rp_vals[:-1] + rp_vals[1:])
                    # first edge: extrapolate using first interval
                    first_edge = rp_vals[0] - (midpoints[0] - rp_vals[0])
                    # last edge: extrapolate using last interval
                    last_edge = rp_vals[-1] + (rp_vals[-1] - midpoints[-1])
                    bounds = np.concatenate(([first_edge], midpoints, [last_edge]))

                # BoundaryNorm ties the discrete colormap to the boundaries
                norm = BoundaryNorm(bounds, ncolors=n_bins, clip=True)

                # plot using discrete colormap and norm
                mappable = ax.pcolormesh(Lon, Lat, values, cmap=discrete_cmap, norm=norm,
                                        shading='auto', transform=crs)

                # colourbar ticks should be exactly at the RP values we want
                rp_ticks = [int(v) for v in rp_vals]

        else:
            # Continuous return level field: use pcolormesh with Normalize
            if np.all(np.isnan(values)):
                # fallback for empty data
                mappable = ax.pcolormesh(Lon, Lat, values, cmap=cmap, shading='auto', transform=crs)
            else:
                vmin = np.nanmin(values)
                vmax = np.nanmax(values)
                norm = Normalize(vmin=vmin, vmax=vmax)
                mappable = ax.pcolormesh(Lon, Lat, values, cmap=cmap, norm=norm,
                                         shading='auto', transform=crs)

    # Title and layout
    title_str = f'{var_name}'
    ax.set_title(title_str, fontsize=16)
    plt.tight_layout()

    # --- Colourbar handling ---
    # Use the mappable created by contourf/pcolormesh for the colorbar so ticks/levels align
    if mappable is None:
        # safety fallback
        mappable = ax.pcolormesh(Lon, Lat, values, cmap=cmap, shading='auto', transform=crs)

    if 'precip' in var_name:
        # For precipitation, we want the colorbar to have discrete levels matching the contourf levels
        cbar = fig.colorbar(mappable, ax=ax, orientation='horizontal', 
                            fraction=0.05, pad=0.04)
        cbar.set_label('Accumulated precip (mm)')
    else:
        cbar = fig.colorbar(mappable, ax=ax, orientation='horizontal', 
                            fraction=0.05, pad=0.04, 
                            boundaries=bounds, spacing='uniform')
        midpoints = 0.5 * (bounds[:-1] + bounds[1:])
        cbar.set_ticks(midpoints)
        cbar.set_ticklabels([str(int(v)) for v in rp_vals])
        cbar.set_label('Return period (years)')

    # Label the colorbar appropriately
    if var_name.endswith('year return levels') or 'return level' in var_name.lower():
        cbar.set_label('Return level (mm)')

    return fig


def identify_top_n_forecasts(da: xr.DataArray,
                             event_str: str,
                             region_extent=[],
                             cmap='viridis',
                             top_num=5):
    """
    Identify the top N (e.g. 5) forecasts in an ensemble and create labels for future plotting 
    Return a list of the selected ensemble members and associated colours from a chosen colourmap for plotting  
    """
    if region_extent is not None:
        lon0, lon1, lat0, lat1 = region_extent
        ds = da.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1)).sum(dim=['lat','lon'])
    else:
        ds = da
    sample_max = ds.max(dim='lead_time')

    if event_str == 'Desmond':
        multi_index = ds['member'].to_index()
        ens_values = multi_index.get_level_values('seed').astype(int)
    elif event_str == 'Desmond_seed76' or event_str == 'Desmond_seed76_v2':
        multi_index = ds['member'].to_index()
        ens_values = multi_index.get_level_values('member').astype(int)
    # TODO: build in more generalised functionality here

    series = pd.Series(sample_max.values, index=ens_values)
    per_seed_max = series.groupby(series.index).max().dropna()
    top_num = 5
    highlight_seeds = per_seed_max.sort_values(ascending=False).index[:top_num].tolist()

    cmap = plt.get_cmap(cmap)
    highlight_seed_colours = {seed: cmap(i/(top_num-1)) for i, seed in enumerate(highlight_seeds)}
    return ds, per_seed_max, highlight_seeds, highlight_seed_colours, multi_index


def produce_postage_stamp_plot(files: list,
                               event_str: str,
                               var='pr24',
                               lead_time=300,
                               panel_size=1.3,
                               ncols=10,
                               region_extent=[],
                               min_max_bbox=[],
                               cmap=get_metpy_colourmap('precipitation'),
                               crs=ccrs.PlateCarree(),
                               lat_name_hints=('lat','latitude'),
                               lon_name_hints=('lon','longitude'),
                               ):
    """
    Produce a postage stamp plot of all ensemble members within a chosen forecast, at a specific lead time (e.g. T+300)
    """

    # pre-processing 
    coast_res = '50m'  # '110m' (faster), '50m' (default), '10m' (slower)

    # forecast lead time (initialised at 00Z on 23 November 2015)
    if event_str == 'non_event':
        best_lead = 144
    else: # e.g. Desmond, Desmond_seed76, ...
        best_lead = lead_time

    # read in and pre-process the nc files for mslp and precip 
    da_pr_all, da_mslp_all = preprocess_input_nc_files(files,
                                                       rolling_window=4,
                                                       min_periods=4,
                                                       event_str=event_str,
                                                       lead_dim='lead_time')

    # ---- detect lat/lon dims ----
    dims = da_pr_all.dims
    lat_dim = next((d for d in dims if any(h in d.lower() for h in lat_name_hints)), None)
    lon_dim = next((d for d in dims if any(h in d.lower() for h in lon_name_hints)), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError("Could not find lat/lon dims. Provide lat_name_hints or rename coords.")

    # select requested lead_time
    best_lead = best_lead
    da_pr_24 = da_pr_all.sel(lead_time=best_lead)
    da_mslp_all = da_mslp_all.sel(lead_time=best_lead)

    # get readable labels using the 'seed' and 'ensemble' information from the index that we've created 
    sample_index = da_mslp_all['member'].to_index()  # Index of (time, ensemble) tuples
    if event_str == 'Desmond_seed76' or event_str == 'Desmond_seed76_v2':
        labels = [f"ens: {ens}" for ens in sample_index]
    else: # any forecast with multiple input nc files ('seed_xx.nc')
        labels = [f"seed: {t}, ens: {e}" for (t, e) in sample_index]

    # TODO: tidy up the code below to keep the option of plotting 6-h precip (even though we likely won't need it)
    # processing + plotting steps  
    da_pr_plot = da_pr_24
    interval_str = '24h'
    # if var == 'pr06':
    #     da_pr_plot = da_pr_06
    #     interval_str = '6h'
    # elif var == 'pr24':
    #     da_pr_plot = da_pr_24
    #     interval_str = '24h'

    # get ensemble coordinate values (labels)
    n_ens = da_pr_plot.sizes['member'] # TODO: double check that this line doesn't refer to the previous version of 'da_pr_all' 

    # grid layout
    nrows = math.ceil(n_ens / ncols)
    fig_w = ncols * panel_size
    fig_h = nrows * panel_size
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             subplot_kw={'projection': crs})
    axes = np.asarray(axes).reshape(-1)  # flatten (works whether grid is 1D or 2D)

    # iterate panels and plot each member
    lon = da_pr_plot['lon'].values # TODO: double check that this line doesn't refer to the previous version of 'da_pr_all' 
    lat = da_pr_plot['lat'].values # TODO: double check that this line doesn't refer to the previous version of 'da_pr_all' 
    Lon, Lat = np.meshgrid(lon, lat)  # for contour plotting if desired

    # ---- restrict score region (UK) ----
    if min_max_bbox is not None:
        lon0, lon1, lat0, lat1 = min_max_bbox
        da_score = da_pr_plot.sel({lat_dim: slice(lat0, lat1), lon_dim: slice(lon0, lon1)})
    else:
        da_score = da_pr_plot

    # compute common contour levels across all members for comparability
    # originally used percentiles to avoid being dominated by outliers but changed to min/max  
    if var == 'pr24' or var == 'pr06':
        vmin = np.min(da_score.values)
        vmax = np.max(da_score.values)
        interval = 1.0
    # TODO: modify code to include mslp as well as precip
    elif var == 'mslp':
        vmin = np.min(da_mslp_all.values)
        vmax = np.max(da_mslp_all.values)
        interval = 2.0
    lev_min = np.floor(vmin / interval) * interval
    lev_max = np.ceil(vmax / interval) * interval
    levels = np.arange(lev_min, lev_max + 1e-6, interval)

    for i in range(n_ens):
        ax = axes[i]
        ax.set_extent(region_extent, crs=crs)  # adjust to desired UK+Ireland extent

        # plot coastlines/land
        ax.coastlines(resolution=coast_res, linewidth=0.4)
        ax.add_feature(cfeature.BORDERS.with_scale(coast_res), linestyle=':', linewidth=0.3)
        ax.add_feature(cfeature.LAND.with_scale(coast_res), facecolor='lightgray', zorder=0)

        # plot either mslp or precip 
        if var == 'mslp':
            da_member = da_mslp_all.isel(member=i)  # select a single ensemble forecast to plot 

            # contour isobars; use thinner lines for small panels
            cs = ax.contour(lon, lat, da_member.values, levels=levels,
                            colors='k', linewidths=0.4, transform=crs)

        elif var == 'pr24' or var == 'pr06':
            da_member = da_pr_plot.isel(member=i)
            cs = ax.contourf(lon, lat, da_member.values, levels=levels, 
                            cmap=cmap, transform=crs, alpha=0.9)

        # small title with ensemble id (keep font small)
        ax.set_title(labels[i], fontsize=7, pad=2)

        # remove tick labels for interior panels to keep things compact
        col = i % ncols
        row = i // ncols
        if col != 0:
            ax.set_yticklabels([])
            ax.set_yticks([])
        if row != (nrows - 1):
            ax.set_xticklabels([])
            ax.set_xticks([])

    # turn off any unused axes (if ncols*nrows > n_ens)
    for j in range(n_ens, len(axes)):
        axes[j].axis('off')

    # overall suptitle and a shared legend/colorbar if desired
    # TODO: edit the if statement below to encompass 'pr24' and 'pr06' without having to explicitly specify each one 
    if var == 'mslp':
        plt.suptitle(f'MSLP (hPa) — lead T+{best_lead} — {event_str}', fontsize=12, y=0.99)
    elif var == 'pr':
        plt.suptitle(f'Precip (mm) — lead T+{best_lead} — {event_str}', fontsize=12, y=0.99)

    # optional shared colorbar: create an invisible mappable with the contour levels
    norm = mcolors.Normalize(vmin=lev_min, vmax=lev_max)
    sm = ScalarMappable(norm=norm, cmap=cmap)  # only used for colorbar scaling if you want
    sm.set_array([])
    plt.tight_layout()

    # place horizontal colorbar below the figure
    if var == 'pr24' or var == 'pr06':
        cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='horizontal', fraction=0.03, pad=0.02)
        cbar.set_label(f'{interval_str} accumulated precip (mm)')
    return fig


def find_and_plot_top_members(
        files: list,               # list of input files (same as `produce_postage_stamp_plot`)
        event_str: str,            # event string ('Desmond', 'Desmond_seed76', 'Desmond_seed76_v2', ...)
        top_k=10,                  # number of panels to include
        score_kind='spatial_max',  # 'spatial_max', 'spatial_mean',  etc.
        select_lead=None,          # if not None, select a single lead (value or index) to plot (e.g. '2015-12-05T00')
        member_stack_dims=None,    # e.g. ('seed','ensemble') or ('time','ensemble') or None to detect or use 'member'
        lat_name_hints=('lat','latitude'),
        lon_name_hints=('lon','longitude'),
        region_extent=[],        # [lon0, lon1, lat0, lat1] - lat0 must be larger than lat1 (grid is oriented S to N)
        ncols=5,
        panel_size=2.2,
        cmap=get_metpy_colourmap('precipitation'),
        crs=ccrs.PlateCarree(),
        add_colourbar=True,
        lead_time=300,
        title_prefix='Most extreme SFNO ensemble members'
):
    """
    Find top_k members by an extremeness score and plot postage-stamp maps of each member.
    da_tp24: DataArray with dims including lat & lon and at least one sampling dimension (member or to be stacked).
    If da_tp24 has separate dims like 'seed' and 'ensemble' (or 'time'+'ensemble'), provide them in member_stack_dims
    (tuple of dim names) and the function will stack -> 'member'.
    """

    # TODO: update this function to ingest different types of input data relevant for specific events ('Desmond', 'Desmond_seed76', 'Desmond_seed76_v2', ...)

    # read in and pre-process the nc files for mslp and precip 
    da_pr_all, _ = preprocess_input_nc_files(files,
                                             rolling_window=4,
                                             min_periods=4,
                                             event_str=event_str,
                                             lead_dim='lead_time')

    # select requested lead_time as before (try label then fallback to isel)
    best_lead = lead_time  # keep variable

    # # calculate 24-h accumulated precip 
    # TODO: tidy this part (no longer need to calculate rolling 24-h total here)
    # da_pr_24 = da_pr_all.sel(lead_time=slice(best_lead,best_lead+23)).sum(dim='lead_time')
    # da_mslp_all = da_mslp_all.sel(lead_time=best_lead)

    # ----- identify lat/lon dims -----
    dims = da_pr_all.dims
    lat_dim = next((d for d in dims if any(h in d.lower() for h in lat_name_hints)), None) # TODO: better understand this type of structure
    lon_dim = next((d for d in dims if any(h in d.lower() for h in lon_name_hints)), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError("Could not find lat/lon dims. Provide lat_name_hints or rename coords.")

    # ----- subset region for scoring if requested -----
    da_score = da_pr_all
    if region_extent is not None:
        lon0, lon1, lat0, lat1 = region_extent
        da_score = da_score.sel({lat_dim: slice(lat0, lat1), lon_dim: slice(lon0, lon1)})

    # ----- stack sampling dims into a single 'member' dim if needed -----
    # If user passed stack dims, use them. Else detect an existing 'member' dim or stack anything except lat/lon and lead_time.
    if member_stack_dims is not None:
        da_sample = da_score.stack(member=member_stack_dims)
    elif 'member' in da_score.dims:
        da_sample = da_score
    else:
        # pick dims that are not spatial (lat, lon). Also allow 'lead_time' to remain if present.
        non_spatial = [d for d in da_score.dims if d not in (lat_dim, lon_dim)]
        # If we have just one non-spatial (e.g. member) that's fine; if multiple, stack them.
        if len(non_spatial) == 1:
            da_sample = da_score
        else:
            da_sample = da_score.stack(member=tuple(non_spatial))

    # Now ensure there is a 'member' dim:
    if 'member' not in da_sample.dims:
        # If da_sample already had a member dimension under another name, rename it:
        # try to detect a dim with many labels that looks like member
        raise ValueError(f"No 'member' dim found after stacking; da_sample dims: {da_sample.dims}")

    # ----- if select_lead provided, restrict to that lead BEFORE scoring or plotting -----
    lead_dim = next((d for d in da_sample.dims if d not in (lat_dim, lon_dim, 'member')), None) # TODO: better understand this structure
    if select_lead is not None and lead_dim is not None:
        try:
            da_sample = da_sample.sel({lead_dim: select_lead})
        except Exception:
            # fallback to numeric index if label selection fails
            da_sample = da_sample.isel({lead_dim: int(select_lead)})

    # ----- compute per-member score -----
    # choose the summary that defines "extreme" member
    if score_kind == 'spatial_max':
        # max across lat & lon (and across lead if present)
        spatial_dims = [d for d in ('lat','latitude','y', lat_dim) if d in da_sample.dims] + \
                       [d for d in ('lon','longitude','x', lon_dim) if d in da_sample.dims]
        spatial_dims = list(dict.fromkeys(spatial_dims))  # TODO: better understand this line 
        # check for any other non-spatial dims (e.g. 'lead_time', but not 'member')
        extra_dims = [d for d in da_sample.dims if d not in spatial_dims + ['member']]
        score_dims = spatial_dims + extra_dims
        member_score = da_sample.max(dim=tuple(score_dims), skipna=True)
    elif score_kind == 'spatial_mean':
        spatial_dims = [lat_dim, lon_dim]
        member_score = da_sample.mean(dim=tuple(spatial_dims), skipna=True)
        # if lead_time also present, could do .max(dim='lead_time') then mean across space - adapt if needed
    else:
        raise ValueError("Unknown score_kind")

    # member_score is a DataArray with dim 'member'; reduce to numpy
    # If member is MultiIndex, get index labels for titles
    member_index = da_sample['member'].to_index() if isinstance(da_sample['member'].to_index(), (np.ndarray,)) else da_sample['member'].to_index()
    # convert to numpy floats
    # If dask-backed, compute
    if hasattr(member_score.data, 'compute'):
        member_score_vals = member_score.compute().values
    else:
        member_score_vals = member_score.values

    # get top_k indices
    order = np.argsort(member_score_vals)[::-1]
    topk_idx = order[:top_k]
    topk_members = [member_index[i] for i in topk_idx]   # these are tuples if MultiIndex

    # ----- Prepare for plotting: pick a single lead_time to plot (if not selected above) -----
    # If select_lead was None and there is lead_dim, choose the lead where the member's score occurred or user can pass.
    if select_lead is None and lead_dim is not None:
        # choose a representative lead to plot (e.g. the lead at which the top member had its spatial max)
        # compute argmax locations in lead dimension for the top member (optional)
        # For simplicity choose the lead where the global max across all members occurs:
        # flatten over member to find lead index of global max
        max_over_sample = da_score.max(dim=('member', lat_dim, lon_dim), skipna=True) if 'member' in da_score.dims else da_score.max(dim=(lat_dim, lon_dim), skipna=True)
        if lead_dim in max_over_sample.dims:
            # find lead label of maximum
            max_lead_label = max_over_sample.argmax(dim=lead_dim).values
            # best to select the lead coordinate where data is maximal:
            try:
                # if argmax returns integer location we can get label
                idx = int(max_over_sample.argmax(dim=lead_dim).values)
                select_lead_label = da_sample[lead_dim].isel({lead_dim: idx}).values
                da_plot_lead = da_sample.sel({lead_dim: select_lead_label})
            except Exception:
                # fallback: pick middle lead
                da_plot_lead = da_sample.isel({lead_dim: da_sample.sizes[lead_dim]//2})
        else:
            da_plot_lead = da_sample
    else:
        # either select_lead provided (already applied above) or there is no lead_dim
        da_plot_lead = da_sample

    # If da_plot_lead still has 'member' dim and lat/lon dims, proceed
    # create plotting grid
    n_plots = min(top_k, da_plot_lead.sizes['member'])
    ncols = min(ncols, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig_w = ncols * panel_size
    fig_h = nrows * panel_size
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), subplot_kw={'projection': ccrs.PlateCarree()})
    axes_flat = np.asarray(axes).reshape(-1)

    # get MetPy colourmap specifically for plotting precipitation
    #cmap = get_metpy_colourmap('precipitation')   # or another name from MetPy colortables

    # find common plotting vmin/vmax (optional): compute pooled vmin/vmax across selected members
    data_for_range = da_plot_lead.isel(member=topk_idx).values  # shape (topk, lat, lon) or with lead removed
    vmin = float(np.nanmin(data_for_range))
    vmax = float(np.nanmax(data_for_range))
    interval = 1
    lev_min = np.floor(vmin / interval) * interval
    lev_max = np.ceil(vmax / interval) * interval
    levels = np.arange(lev_min, lev_max + 1e-6, interval)

    lon = da_plot_lead[lon_dim].values
    lat = da_plot_lead[lat_dim].values
    Lon, Lat = np.meshgrid(lon, lat)

    # get readable labels using the 'seed' and 'ensemble' information from the index that we've created 
    #sample_index = da_tp24['member'].to_index()  # Index of (time, ensemble) tuples
    labels = [f"seed: {t}, ens: {e}" for (t, e) in topk_members]


    for i, m_idx in enumerate(topk_idx):
        ax = axes_flat[i]
        # select member by integer position
        da_member = da_plot_lead.isel(member=m_idx)

        cs = ax.contourf(lon, lat, da_member.values, levels=levels, 
                         cmap=cmap, transform=crs, alpha=0.9)

        # minimal map decoration
        ax.coastlines(resolution='50m', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidth=0.3)
        ax.set_extent(region_extent if region_extent is not None else [lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
        # title with member label (multiindex tuple -> nice string)
        lab = member_index[m_idx]
        # if lab is a tuple from MultiIndex, join to readable string:
        if isinstance(lab, tuple):
            lab_str = '/'.join([str(x) for x in lab])
        else:
            lab_str = str(lab)
        ax.set_title(labels[i], fontsize=8, pad=2)

    # turn off unused axes
    for j in range(n_plots, len(axes_flat)):
        axes_flat[j].axis('off')

    # shared colorbar: create an invisible mappable with the contour levels
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)  # colourbar scaling 
    sm.set_array([])
    plt.tight_layout()

    # place horizontal colorbar below the figure
    if add_colourbar:
        cbar = fig.colorbar(sm, ax=axes_flat[:n_plots].tolist(), orientation='horizontal', fraction=0.08, pad=0.02)
        cbar.set_label('24-h accumulated precip (mm)')

    fig.suptitle(f"{title_prefix} (top {top_k}): T+{lead_time}", fontsize=12)
    return topk_members, fig, da_plot_lead


def top_members_by_rolling_max(
    files: list,                         # DataArray of precipitation with dims including lead_dim, lat, lon, and sampling dims
    event_str: str,                      # event string ('Desmond', 'Desmond_seed76', 'Desmond_seed76_v2', ...)
    sampling_dims='member',              # dims to stack into a single 'member' (tuple) or a single member dim name
    lead_dim_hint='lead_time',           # name of lead-time dim (string) or None to auto-detect
    lat_name_hints=('lat','latitude'),
    lon_name_hints=('lon','longitude'),
    rolling_window=4,                    # number of lead steps to sum for 24h (4 * 6h = 24h)
    min_periods=4,                       # min_periods for rolling
    score_bbox=None,                     # bbox [lon_min, lon_max, lat_min, lat_max] used for scoring (Dublin)
    plot_bbox=None,                      # bbox used for plotting (UK+Ireland); if None, uses full DA extent
    score_method='sum',                  # method to reduce spatial box to a single score: 'sum', 'mean', or 'max'
    exclusion_window=2,                  # number of lead time steps to exclude around the maximizing lead for each member (avoid double counting)
    top_k=10,
    cmap=get_metpy_colourmap('precipitation'),
    crs=ccrs.PlateCarree(),
    ncols=5,
    panel_size=2.4,
    compute_now=True,                    # set False to manage dask compute externally (only relevant for very large arrays)
    peak_window_hours=None,              # if not None, run the peak-centred-window aggregation across all members
    plot_distribution=False,             # if True and peak_window_hours is not None, also plot a histogram of the aggregated distribution
):
    """
    Find top_k members by maximum rolling-window total inside score_bbox and plot the full-domain field
    at each member's maximizing lead.

    Return: (top_members, best_lead_indices, fig, da_tp24)
    - top_members: list of member labels (MultiIndex tuples if stack used)
    - best_lead_indices: integer positions (in lead dim) where each member's max occurred
    - fig: figure handle
    - da_tp24: DataArray of rolling sums (member, lead, lat, lon)
    """

    # read in and pre-process the nc files for mslp and precip 
    da_pr_all, _ = preprocess_input_nc_files(files,
                                             rolling_window=rolling_window,
                                             min_periods=min_periods,
                                             event_str=event_str,
                                             bbox=plot_bbox,
                                             lead_dim='lead_time')


    # ---- detect lat/lon dims ----
    dims = da_pr_all.dims
    lat_dim = next((d for d in dims if any(h in d.lower() for h in lat_name_hints)), None)
    lon_dim = next((d for d in dims if any(h in d.lower() for h in lon_name_hints)), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError("Could not find lat/lon dims. Provide lat_name_hints or rename coords.")

    # ---- detect or set lead dim ----
    if lead_dim_hint and lead_dim_hint in da_pr_all.dims:
        lead_dim = lead_dim_hint
    else:
        # pick the non-spatial dims and choose one that looks like lead_time
        non_spatial = [d for d in da_pr_all.dims if d not in (lat_dim, lon_dim)]
        if len(non_spatial) == 0:
            lead_dim = None
        else:
            # choose the first non-spatial dim of length > 1 (prefer 'lead_time' if present)
            lead_dim = next((d for d in non_spatial if d.lower().startswith('lead') or d.lower().startswith('step') or d.lower().startswith('time')), non_spatial[0])

    # ---- stack sampling dims into 'member' if needed ----
    if isinstance(sampling_dims, (tuple, list)) and len(sampling_dims) > 1:
        da_s = da_pr_all.stack(member=tuple(sampling_dims))
    elif isinstance(sampling_dims, str) and sampling_dims in da_pr_all.dims:
        # single existing member dim
        da_s = da_pr_all.rename({sampling_dims: 'member'})
    else:
        # try existing 'member' dim or stack all non spatial/lead dims
        if 'member' in da_pr_all.dims:
            da_s = da_pr_all
        else:
            non_spatial = [d for d in da_pr_all.dims if d not in (lat_dim, lon_dim)]
            if lead_dim is not None and lead_dim in non_spatial:
                non_spatial.remove(lead_dim)
            if len(non_spatial) == 0:
                raise ValueError("No sampling dims found to create 'member'. Provide sampling_dims.")
            da_s = da_pr_all.stack(member=tuple(non_spatial))

    # ensure 'member' in dims
    if 'member' not in da_s.dims:
        raise ValueError("Failed to create 'member' dim.")

    # ---- rename the xr.da (no longer need the rolling calculation here) ----
    da_tp24 = da_s

    # optionally compute now (dask)
    if compute_now:
        try:
            da_tp24 = da_tp24.compute()
        except Exception:
            pass

    # ---- restrict score region (Dublin) ----
    if score_bbox is not None:
        lon0, lon1, lat0, lat1 = score_bbox
        da_score = da_tp24.sel({lat_dim: slice(lat0, lat1), lon_dim: slice(lon0, lon1)})
    else:
        da_score = da_tp24

    # compute spatial reduction (member x lead)
    if score_method == 'sum':
        spatial_reduced = da_score.sum(dim=(lat_dim, lon_dim), skipna=True)
    elif score_method == 'mean':
        spatial_reduced = da_score.mean(dim=(lat_dim, lon_dim), skipna=True)
    elif score_method == 'max':
        spatial_reduced = da_score.max(dim=(lat_dim, lon_dim), skipna=True)
    else:
        raise ValueError("score_method must be one of 'sum','mean','max'")

    # spatial_reduced: DataArray with dims ('member', lead_dim) (or vice-versa)
    # ensure lead_dim variable exists from earlier code

    # If spatial_reduced has the lead dim, allow multiple entries per member
    if lead_dim in spatial_reduced.dims:
        # reindex to ensure order ('member', lead_dim) so unravel_index maps correctly
        spatial_rl = spatial_reduced.transpose('member', lead_dim)
    else: # make a synthetic lead_dim of length 1 to unify the logic (member, lead_dim=0)
        spatial_rl = spatial_reduced.expand_dims({lead_dim: [0]}).transpose('member', lead_dim)

    # get numpy array and shape 
    try:
        arr = spatial_rl.values  # shape (n_member, n_lead)
    except Exception:
        arr = spatial_rl.compute().values
    n_member, n_lead = arr.shape


    # If peak_window_hours is provided, perform "peak per member -> centered window -> aggregate" workflow
    #dist_array = None
    fig_dist = None

    if 'peak_window_hours' in locals() and peak_window_hours is not None:
        # Determine lead spacing in hours robustly
        lead_vals = spatial_rl[lead_dim].values
        step_hours = 1.0
        if len(lead_vals) >= 2:
            try:
                # datetime-like or timedelta-like arrays:
                if np.issubdtype(lead_vals.dtype, np.datetime64) or np.issubdtype(lead_vals.dtype, np.timedelta64):
                    # convert to ns ints and compute hour spacing
                    diffs_ns = np.diff(lead_vals.astype('datetime64[ns]').astype('int64'))
                    step_hours = abs(diffs_ns[0]) / (1e9 * 3600.0)
                else:
                    # numeric: assume numeric spacing is in hours or consistent units
                    step_hours = float(np.diff(lead_vals.astype(float))[0])
            except Exception:
                step_hours = 1.0
        # derive number of lead steps to cover the requested hours window
        total_hours = float(peak_window_hours)
        steps_total = max(1, int(round(total_hours / step_hours)))
        half_steps = steps_total // 2

        # find per-member peak index using the bbox-based score (arr)
        peak_indices = np.full(n_member, -1, dtype=int)
        for m in range(n_member):
            row = arr[m, :]
            if np.all(~np.isfinite(row)):
                peak_indices[m] = -1
            else:
                tmp = np.where(np.isfinite(row), row, -np.inf)
                peak_indices[m] = int(np.nanargmax(tmp))

        # Collect full-domain values for centered windows for each member
        collected_full = [] # holds full flattened blocks, included NaNs
        blocks = [] # metadata to reconstruct mapped values later 
        collected_finite_count = 0
        #skipped_members = 0

        for m in range(n_member):
            pidx = peak_indices[m]
            if pidx < 0:
                #skipped_members += 1
                continue
            start = max(0, pidx - half_steps)
            end = min(n_lead, pidx + half_steps + 1)  # end exclusive

            # extract the window for this member from the 24-h totals da_tp24
            try:
                window_da = da_tp24.isel(member=int(m)).isel({lead_dim: slice(start, end)})
            except Exception:
                window_da = da_tp24.sel(member=da_tp24['member'][int(m)]).isel({lead_dim: slice(start, end)})

            if compute_now:
                try:
                    window_da = window_da.compute()
                except Exception:
                    pass

            # flatten lead x lat x lon -> 1D and keep finite values only
            vals = window_da.values
            #flat = vals.ravel()
            flat_full = vals.ravel().astype(np.float32)

            # record metadata for reconstruction
            block = {
                'member': int(m),
                'start': int(start),
                'end': int(end),  # exclusive
                'shape': vals.shape,
                'length': flat_full.size
            }
            blocks.append(block)
            collected_full.append(flat_full)

            # book keeping for the finite-only pooled distribution used for plotting + diagnostics 
            finite_count = np.count_nonzero(np.isfinite(flat_full))
            collected_finite_count += int(finite_count)

            #finite_mask = np.isfinite(flat)
            # if finite_mask.any():
            #     collected.append(flat[finite_mask].astype(np.float32))
            # else:
            #     skipped_members += 1

        # concatenate full pooled vector (includes NaNs)
        if len(collected_full) == 0:
            #raise ValueError("No valid grid-cell values collected across member windows. Check data and windows.")
            raise ValueError("No windows collected across members; check data and parameters.")
        sfno_vals_full = np.concatenate(collected_full)

        # create finite-only pooled distribution for plotting + diagnostics 
        sfno_dist = sfno_vals_full[np.isfinite(sfno_vals_full)]
        #dist_array = np.concatenate(collected)

        # # optional subsampling to limit memory/work for plotting/processing
        # N_max = 2_000_000
        # if dist_array.size > N_max:
        #     idx_sub = np.random.choice(dist_array.size, size=N_max, replace=False)
        #     dist_array = dist_array[idx_sub]

        # optional histogram plot
        if 'plot_distribution' in locals() and plot_distribution:
            fig_dist, axd = plt.subplots(figsize=(6, 4))
            axd.hist(sfno_dist, bins=100, density=False, color='C0', alpha=0.8)
            axd.set_title(f"Aggregated distribution: {len(blocks)} members, {sfno_dist.size} finite samples")
            axd.set_xlabel('24-h total (same units as DA)')
            axd.set_ylabel('Counts')
            plt.tight_layout()
        else:
            fig_dist = None

        # prepare the outputs; return the pooled full vector (with NaNs) and the blocks metadata,
        # plus a finite-only sfno_dist for plotting compatibility with existing code 
        try:
            member_index = da_s['member'].to_index()
            top_members = [member_index[int(i)] for i in np.arange(n_member)[peak_indices >= 0]]
        except Exception:
            top_members = [int(i) for i in np.arange(n_member)[peak_indices >= 0]]
        top_lead_positions = peak_indices[peak_indices >= 0]

        # return signature for peak-window aggregation mode
        return top_members, top_lead_positions, None, da_tp24, sfno_vals_full, blocks, fig_dist

        # PREVIOUS VERSION OF CODE (NO NANS)
        # # construct top_members and top_lead_positions arrays for return compatibility
        # member_idxs = np.arange(n_member)[peak_indices >= 0]
        # try:
        #     member_index = da_s['member'].to_index()
        #     top_members = [member_index[int(i)] for i in member_idxs]
        # except Exception:
        #     top_members = [int(i) for i in member_idxs]
        # top_lead_positions = peak_indices[peak_indices >= 0]

        # # Early return for peak-window aggregation mode. Keep da_tp24 for caller.
        # # Return signature in this mode:
        # # (top_members, top_lead_positions, None, da_tp24, dist_array, fig_dist)
        # return top_members, top_lead_positions, None, da_tp24, dist_array, fig_dist

    # original code is below (if we don't want to find the peak window for all ensemble members)
    else:

        # flatten and sort descending 
        flat = arr.ravel()
        flat_finite = np.where(np.isfinite(flat), flat, -np.inf)

        # build a list sorted by value (descending) with corresponding member and lead indices
        order_flat = np.argsort(flat_finite)[::-1]  # indices of sorted entries in flattened array
        order_flat = order_flat[flat_finite[order_flat] > -np.inf]  # filter out -inf entries (non-finite)

        # map flattened indices back to 2D (member_idx, lead_idx)
        member_idxs_all, lead_idxs_all = np.unravel_index(order_flat, (n_member, n_lead))
        scores_all = flat_finite[order_flat]

        # select top_k entries ensuring no lead time overlap within exclusion_window
        selected = []
        for mem_idx, lead_idx, score in zip(member_idxs_all, lead_idxs_all, scores_all):
            if score == -np.inf:
                continue  # skip non-finite
            # check if this lead_idx is too close to any already selected for the same member]
            already = False
            for (sel_mem_idx, sel_lead_idx, _) in selected:
                if sel_mem_idx == mem_idx and abs(sel_lead_idx - lead_idx) <= exclusion_window:
                    already = True
                    break
            if not already:
                selected.append((int(mem_idx), int(lead_idx), float(score)))
            if len(selected) >= top_k:
                break

        # below instance is unlikely but is useful as a worst case check to avoid proceeding with empty selections
        if len(selected) == 0:
            raise ValueError("No valid candidates found for selection.")

        # extract member indices and lead positions from selected tuples
        member_idxs = np.array([s[0] for s in selected], dtype=int)
        lead_idxs = np.array([s[1] for s in selected], dtype=int)
        top_values = np.array([s[2] for s in selected], dtype=float)

        # map to labels if available (member_index may be MultiIndex)
        try:
            member_index = da_s['member'].to_index()
        except Exception:
            member_index = np.arange(da_s.sizes['member'])
        top_members = [member_index[int(i)] for i in member_idxs]
        top_lead_positions = lead_idxs  # integer positions relative to lead_dim

        # ---- prepare plotting: for each selected member pick its lead and plot full domain ----
        selected_fields = []
        display_values = []
        # common vmin/vmax across selected members (full-domain fields)
        for mem_i, lead_i in zip(member_idxs, lead_idxs):
            # select member and lead by integer positions
            if lead_dim in da_tp24.dims:
                field = da_tp24.isel(member=int(mem_i)).isel({lead_dim: int(lead_i)})
            else:
                field = da_tp24.isel(member=int(mem_i))
            # choose plotting bbox
            if plot_bbox is not None:
                lon0p, lon1p, lat0p, lat1p = plot_bbox
                field_plot = field.sel({lat_dim: slice(lat0p, lat1p), lon_dim: slice(lon0p, lon1p)})
            else:
                field_plot = field
            selected_fields.append(field_plot)
            # compute a per-gridbox max/mean to display in title if needed
            display_values.append(float(field.max(dim=(lat_dim, lon_dim), skipna=True).values))

        # stack to compute common vmin/vmax
        stacked_sel = xr.concat([f for f in selected_fields], dim='stack_temp')
        vmin = float(stacked_sel.min().values)
        vmax = float(stacked_sel.max().values)

        # panel layout
        n_plots = len(selected)
        ncols = min(ncols, n_plots)
        nrows = math.ceil(n_plots / ncols)
        fig_w = ncols * panel_size
        fig_h = nrows * panel_size
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        axes = np.array(axes).reshape(-1)

        # contour levels
        interval = 1
        lev_min = np.floor(vmin / interval) * interval
        lev_max = np.ceil(vmax / interval) * interval
        levels = np.arange(lev_min, lev_max + 1e-6, interval)

        lon = da_s[lon_dim].values
        lat = da_s[lat_dim].values

        """
        - 'ax_i' = panel index (0 to 9), 'm_idx' = ensemble member index (0 to 127), 'lead_pos' = lead time index (0 to 59)
        'zip' function pairs two lists to create a tuple 
        'enumerate' function wraps the zipped pairs so that each pair also gets a sequential index number 
        """

        # Loop over the selected top members and their corresponding lead positions to plot each one in a subplot
        for ax_i, (m_idx, lead_pos, score, m_lbl) in enumerate(zip(member_idxs, lead_idxs, top_values, top_members)):
            ax = axes[ax_i]
            if lead_dim in da_tp24.dims:
                field = da_tp24.isel(member=int(m_idx)).isel({lead_dim: int(lead_pos)})
            else:
                field = da_tp24.isel(member=int(m_idx))
            #pcm = ax.pcolormesh(lon, lat, field.values, transform=ccrs.PlateCarree(),
            #                    cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        
            cs = ax.contourf(lon, lat, field.values, levels=levels, 
                            cmap=cmap, transform=crs, alpha=0.9)
            ax.coastlines(resolution='50m', linewidth=0.5)
            ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidth=0.3)
            if plot_bbox is not None:
                ax.set_extent([lon0p, lon1p, lat0p, lat1p], 
                            crs=ccrs.PlateCarree())
            else:
                ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], 
                            crs=ccrs.PlateCarree())
            
            # make a descriptive title: member label and the lead label if available
            member_label = member_index[int(m_idx)]

            # robust unpacking/formatting:
            if isinstance(member_label, tuple):
                # typical case when member is a MultiIndex (e.g. (seed, ensemble))
                seed, ens = (int(x) for x in member_label)
                label_str = f"seed: {seed}, ens: {ens}"
            else:
                # not a tuple: might be a numpy scalar (np.int64) or string or single-level index
                try:
                    seed = int(member_label)      # works for np.int64 and normal ints
                    # no ensemble level available
                    label_str = f"seed: {seed}"
                except Exception:
                    # fallback: stringify and try to parse if it contains both parts, e.g. "seed26/ens3"
                    s = str(member_label)
                    if '/' in s:
                        a, b = s.split('/', 1)
                        label_str = f"seed: {a.strip()}, ens: {b.strip()}"
                    else:
                        label_str = s

            lead_coord = da_tp24[lead_dim].values[int(lead_pos)]
            if isinstance(member_label, tuple):
                m_lbl = f"seed {seed}, ens {ens}"
            else:
                m_lbl = label_str
            title = f"{m_lbl}:  lead time T+{int(lead_coord)}"
            ax.set_title(title, fontsize=8)

        # turn off unused axes
        for j in range(n_plots, len(axes)):
            axes[j].axis('off')

        # shared colorbar: create an invisible mappable with the contour levels
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = ScalarMappable(norm=norm, cmap=cmap)  # colourbar scaling 
        sm.set_array([])
        plt.tight_layout()

        # place horizontal colorbar below the figure
        cbar = fig.colorbar(sm, ax=axes[:n_plots].tolist(), orientation='horizontal', fraction=0.08, pad=0.03)
        cbar.set_label('24-h accumulated precip (mm)')
        return top_members, top_lead_positions, fig, da_tp24