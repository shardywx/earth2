import plotting_functions as pf
import xarray as xr
import numpy as np
import pandas as pd

def preprocess_era5_rainfall(ds: xr.Dataset,
                             rolling_window: int,
                             min_periods: int,
                             event_end_date = '2015-12-06T00',
                             bbox = [-12.0, 2.5, 48.0, 60.0],
                             lead_dim = 'valid_time') -> xr.DataArray:
    """ 
    Pre-process ERA5 rainfall data to be in the correct format for calculating threshold exceedance. This includes:
        - converting units from m to mm
    TODO: create the correct file structure to import 'plotting_functions' (in the same directory)
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
    ds, _, lon_name = pf.select_subset_by_bbox(ds, bbox=bbox)

    # normalise longitude if required (e.g. if dataset uses 0..360 but we want -180..180 for plotting)
    ds = pf.normalise_longitude(ds, lon_name=lon_name)

    # change units of precip from 'm' to 'mm'
    # TODO: generalise for different variable names and units, e.g. by passing in the variable name and expected input and output units as parameters
    da_pr = ds['tp'] * 1000.0

    # calculate rolling totals if required (e.g. to get 24-h totals from hourly data)
    da_pr_roll = pf.calculate_rolling_total(da_pr, 
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


def da_grid_to_point_df(da: xr.DataArray,
                        lat_names=('lat', 'latitude', 'y'),
                        lon_names=('lon', 'longitude', 'x'),
                        value_name='return_period',
                        dropna=True) -> pd.DataFrame:
    """
    Convert a 2D xarray.DataArray (lat x lon) into a pandas DataFrame with columns:
      ['latitude', 'longitude', value_name]
    Handle common lat/lon coord names and both 1D (preferred) and 2D coordinate arrays.
    If dropna=True, rows where value is NaN are removed (useful to drop ocean cells).
    """
    # find lat/lon coordinate names (coords preferred, then dims)
    lat_coord = next((c for c in da.coords if c.lower() in lat_names), None)
    lon_coord = next((c for c in da.coords if c.lower() in lon_names), None)
    if lat_coord is None:
        lat_coord = next((d for d in da.dims if d.lower() in lat_names), None)
    if lon_coord is None:
        lon_coord = next((d for d in da.dims if d.lower() in lon_names), None)

    # fallback: if dims themselves are exactly two and coords not found, use them
    if lat_coord is None or lon_coord is None:
        if len(da.dims) == 2:
            lat_coord, lon_coord = da.dims[0], da.dims[1]
        else:
            raise ValueError("Could not determine lat/lon coordinate names from DataArray.")

    # extract coordinate arrays
    lat_vals = da.coords.get(lat_coord, None)
    lon_vals = da.coords.get(lon_coord, None)

    if lat_vals is None or lon_vals is None:
        # fallback to use index positions
        nlat = da.sizes[da.dims[0]]
        nlon = da.sizes[da.dims[1]]
        Lon, Lat = np.meshgrid(np.arange(nlon), np.arange(nlat))
    else:
        lat_arr = np.asarray(lat_vals)
        lon_arr = np.asarray(lon_vals)

        # if both are 1D arrays (lat, lon) -> make meshgrid
        if lat_arr.ndim == 1 and lon_arr.ndim == 1:
            Lon, Lat = np.meshgrid(lon_arr, lat_arr)
        # if both are 2D arrays matching the DataArray grid, use them directly
        elif lat_arr.ndim == 2 and lon_arr.ndim == 2 and lat_arr.shape == da.shape and lon_arr.shape == da.shape:
            Lat = lat_arr
            Lon = lon_arr
        # if one is 1D and the other is 2D or shapes mismatch, attempt to broadcast sensibly:
        else:
            # try to index coordinate by dims to obtain per-point arrays
            try:
                Lat = da.coords[lat_coord].values
                Lon = da.coords[lon_coord].values
                if Lat.shape != da.shape or Lon.shape != da.shape:
                    # last resort: construct from 1D pairs if possible
                    if Lat.ndim == 1 and Lon.ndim == 1:
                        Lon, Lat = np.meshgrid(Lon, Lat)
                    else:
                        raise ValueError("Coordinate arrays shapes don't match data shape.")
            except Exception as e:
                raise ValueError("Unable to interpret lat/lon coordinates: " + str(e))

    # flatten values in a consistent (row-major) order: lat first then lon
    values = da.values
    if values.shape != Lat.shape:
        # try to transpose if dims order differs
        try:
            # attempt to reorder da to match (Lat dim order)
            # find dimension names that correspond to Lat rows/cols
            values = da.transpose(*da.dims).values
        except Exception:
            raise ValueError(f"DataArray shape {values.shape} does not match coordinate grid shape {Lat.shape}.")

    flat_lat = Lat.ravel()
    flat_lon = Lon.ravel()
    flat_val = values.ravel()

    df = pd.DataFrame({
        'latitude': flat_lat,
        'longitude': flat_lon,
        value_name: flat_val
    })

    if dropna:
        df = df[np.isfinite(df[value_name])].reset_index(drop=True)

    return df


def map_event_to_return_period(event_da: xr.DataArray,
                               thresholds_da: xr.DataArray,
                               rp_dim: str | None = None,
                               acc_dim: str | None = None,
                               acc_value: float | int | None = None,
                               fill_value=np.nan,
                               bbox: list | None = None) -> xr.DataArray:
    """
    Map each grid point of event_da to the smallest return period whose threshold >= event value.

    Robust to differing lat/lon dim/coord names, masked arrays, and thresholds that include
    singleton dims (these will be squeezed). If thresholds have non-singleton extra dims
    (e.g. multiple accumulation periods) the function will raise and ask you to select the
    appropriate acc_dim / reduce thresholds_da before calling.
    """

    # helper sets for lat/lon name detection
    lat_name_candidates = {'lat', 'latitude', 'y', 'projection_y_coordinate'}
    lon_name_candidates = {'lon', 'longitude', 'x', 'projection_x_coordinate'}

    # --- find rp_dim if not provided ---
    if rp_dim is None:
        possible_rp_names = ['return_level', 'rl', 'return_period', 'rp', 'rp_years',
                             'return_period_years', 'return_periods']
        found = [name for name in possible_rp_names if name in thresholds_da.dims or name in thresholds_da.coords]
        if not found:
            raise ValueError("Could not determine return period dim. Please provide rp_dim.")
        rp_dim = found[0]

    # --- handle accumulation dimension selection if requested ---
    if acc_value is not None:
        if acc_dim is None:
            possible_acc = ['accumulation_period', 'acc', 'accumulation', 'period_hours']
            found_acc = [name for name in possible_acc if name in thresholds_da.dims or name in thresholds_da.coords]
            if found_acc:
                acc_dim = found_acc[0]
        if acc_dim is not None:
            if acc_dim not in thresholds_da.dims and acc_dim not in thresholds_da.coords:
                raise ValueError(f"acc_dim {acc_dim} not found in thresholds_da")
            thresholds_da = thresholds_da.sel({acc_dim: acc_value})

    # ensure rp coords ascending
    if rp_dim not in thresholds_da.coords and rp_dim not in thresholds_da.dims:
        raise ValueError(f"rp_dim '{rp_dim}' not found in thresholds_da dims or coords.")
    rp_coords = thresholds_da[rp_dim].values
    if not np.all(np.diff(rp_coords) >= 0):
        thresholds_da = thresholds_da.sortby(rp_dim)
        rp_coords = thresholds_da[rp_dim].values

    # --- Squeeze any singleton dims in thresholds_da that are not rp_dim (safe) ---
    extra_dims = [d for d in thresholds_da.dims if d != rp_dim]
    for d in list(extra_dims):
        if thresholds_da.sizes.get(d, 1) == 1:
            thresholds_da = thresholds_da.squeeze(d, drop=True)
            extra_dims.remove(d)

    # # --- Optional: subset event_da to bbox if provided. Detect lat/lon names robustly ---
    # def detect_lat_lon_names(da: xr.DataArray):
    #     lat_name = next((c for c in da.coords if c.lower() in lat_name_candidates), None)
    #     lon_name = next((c for c in da.coords if c.lower() in lon_name_candidates), None)
    #     if lat_name is None:
    #         lat_name = next((d for d in da.dims if d.lower() in lat_name_candidates), None)
    #     if lon_name is None:
    #         lon_name = next((d for d in da.dims if d.lower() in lon_name_candidates), None)
    #     return lat_name, lon_name

    # lat_name_ev, lon_name_ev = detect_lat_lon_names(event_da)

    # if bbox is not None:
    #     if lat_name_ev is None or lon_name_ev is None:
    #         # fallback: try thresholds coords
    #         lat_name_thr = next((c for c in thresholds_da.coords if c.lower() in lat_name_candidates), None)
    #         lon_name_thr = next((c for c in thresholds_da.coords if c.lower() in lon_name_candidates), None)
    #         if lat_name_ev is None:
    #             lat_name_ev = lat_name_thr
    #         if lon_name_ev is None:
    #             lon_name_ev = lon_name_thr
    #     if lat_name_ev is None or lon_name_ev is None:
    #         raise ValueError("Cannot apply bbox: could not find lat/lon coord names in event_da or thresholds_da.")
    #     lon0, lon1, lat0, lat1 = bbox
    #     try:
    #         lat_vals = event_da[lat_name_ev].values
    #         # slice with correct ordering depending on whether lat is ascending
    #         if np.size(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
    #             event_da = event_da.sel({lat_name_ev: slice(lat1, lat0), lon_name_ev: slice(lon0, lon1)})
    #         else:
    #             event_da = event_da.sel({lat_name_ev: slice(lat0, lat1), lon_name_ev: slice(lon0, lon1)})
    #     except Exception:
    #         # if event_da selection fails, try to use thresholds coords (but silently continue if selection impossible)
    #         pass

    # use our existing bounding box selection function 
    if bbox is not None:
        event_da, _, _ = pf.select_subset_by_bbox(event_da, 
                                                  bbox=bbox)

    # --- Determine spatial dims robustly ---
    thr_spatial = [d for d in thresholds_da.dims if d != rp_dim and d != acc_dim]

    # prefer dims present in both thresholds and event
    spatial_dims = [d for d in thr_spatial if d in event_da.dims]

    # try to infer lat/lon from event_da dims if none found
    if not spatial_dims:
        lat_dim = next((d for d in event_da.dims if d.lower() in lat_name_candidates), None)
        lon_dim = next((d for d in event_da.dims if d.lower() in lon_name_candidates), None)
        if lat_dim and lon_dim:
            spatial_dims = [lat_dim, lon_dim]

    # fallback: if event_da already has exactly 2 dims, use them
    if not spatial_dims and len(event_da.dims) == 2:
        spatial_dims = list(event_da.dims)

    if not spatial_dims or len(spatial_dims) < 2:
        raise ValueError(f"Unable to determine spatial dims. event_da.dims={event_da.dims}, thresholds_da.dims={thresholds_da.dims}")

    # ensure consistent ordering: prefer event_da order for spatial dims
    event_spatial_order = [d for d in event_da.dims if d in spatial_dims]
    if len(event_spatial_order) == 2:
        final_spatial_dims = event_spatial_order
    else:
        final_spatial_dims = list(spatial_dims)

    # --- NEW: if thresholds spatial dims have different names than event, rename thresholds dims to match event dims ---
    thr_spatial_current = [d for d in thresholds_da.dims if d != rp_dim and d != acc_dim]
    if len(thr_spatial_current) == len(final_spatial_dims):
        # If names differ but lengths match, rename thresholds dims to event dims in the event order.
        if tuple(thr_spatial_current) != tuple(final_spatial_dims):
            rename_map = {thr: ev for thr, ev in zip(thr_spatial_current, final_spatial_dims)}
            try:
                thresholds_da = thresholds_da.rename(rename_map)
                # recompute rp_coords in case rename affected ordering (rp_dim unaffected)
                rp_coords = thresholds_da[rp_dim].values
            except Exception:
                # If rename fails, continue and let validation catch the mismatch
                pass

    # --- Reduce any extra dims in event_da (only choose dims with size>1 to avoid making 0-D) ---
    other_dims = [d for d in event_da.dims if d not in final_spatial_dims]
    if other_dims:
        isel_dict = {}
        for d in other_dims:
            try:
                if event_da.sizes.get(d, 1) > 1:
                    isel_dict[d] = 0
            except Exception:
                continue
        if isel_dict:
            event_da = event_da.isel(isel_dict)

    # After reduction, ensure event_da is 2-D
    if event_da.ndim != 2:
        raise ValueError(f"event_da must be 2-D after reduction to spatial dims; got dims {event_da.dims} and ndim {event_da.ndim}.")

    # reorder event_da to final_spatial_dims if necessary
    if tuple(final_spatial_dims) != tuple(event_da.dims):
        try:
            event_da = event_da.transpose(*final_spatial_dims)
        except Exception:
            final_spatial_dims = list(event_da.dims)

    # --- Convert masked arrays to plain numpy with NaNs so comparisons behave predictably ---
    ev_vals = event_da.values
    if np.ma.isMaskedArray(ev_vals):
        ev_vals = ev_vals.filled(np.nan)
        event_da = xr.DataArray(ev_vals, coords={d: event_da[d].values for d in event_da.dims}, dims=event_da.dims)

    # --- Validate thresholds_da dims: squeeze any remaining singleton dims and ensure no unexpected dims ---
    allowed_dims = {rp_dim} | set(final_spatial_dims)
    # Squeeze length-1 dims (except rp_dim) if any still present
    for d in list(thresholds_da.dims):
        if d == rp_dim:
            continue
        if thresholds_da.sizes.get(d, 1) == 1:
            thresholds_da = thresholds_da.squeeze(d, drop=True)
    # Recompute unexpected dims
    unexpected = set(thresholds_da.dims) - allowed_dims
    if unexpected:
        raise ValueError(
            "thresholds_da contains unexpected non-singleton dimension(s) not matching rp_dim + spatial dims: "
            f"{sorted(list(unexpected))}. Please select the appropriate accumulation/other dimension "
            "(e.g. acc_dim) or reduce thresholds_da to dims (rp_dim, lat, lon) before calling."
        )

    # --- Reindex thresholds to event coords (only on matching dims) ---
    reindex_dict = {}
    for d in thresholds_da.dims:
        if d == rp_dim:
            continue
        if d in event_da.coords:
            reindex_dict[d] = event_da[d].values
    if reindex_dict:
        thresholds_da = thresholds_da.reindex(reindex_dict, method='nearest')

    # --- Expand event along rp_dim and align for vectorised comparison ---
    event_exp = event_da.expand_dims({rp_dim: thresholds_da[rp_dim].values})
    thresholds_da, event_exp = xr.align(thresholds_da, event_exp, join='inner')

    # --- Create boolean mask of thresholds >= event ---
    mask = thresholds_da >= event_exp

    # --- For each grid cell, determine whether any RP satisfies threshold >= event and get first RP index ---
    any_true = mask.any(dim=rp_dim)
    idx = mask.argmax(dim=rp_dim).astype(int)

    # --- Build numeric result array ---
    idx_vals = idx.values
    any_true_vals = any_true.values.astype(bool)
    result_arr = np.full(idx_vals.shape, fill_value, dtype=float)

    if any_true_vals.any():
        flat_idx = idx_vals[any_true_vals]
        rp_coords_arr = np.asarray(rp_coords)
        try:
            result_arr[any_true_vals] = rp_coords_arr[flat_idx]
        except Exception:
            result_arr[any_true_vals] = rp_coords_arr[np.asarray(flat_idx, dtype=int)]

    # --- Build coords for result using event_da coords for final_spatial_dims ---
    coords_for_result = {}
    for d in final_spatial_dims:
        if d in event_da.coords:
            coords_for_result[d] = event_da[d].values
        else:
            coords_for_result[d] = np.arange(event_da.sizes[d])

    result_da = xr.DataArray(result_arr,
                             coords=coords_for_result,
                             dims=final_spatial_dims,
                             name="mapped_return_period")

    return result_da