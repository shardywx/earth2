#!/usr/bin/env python3
"""
ERA5 UK/IE Return Levels — Threaded, Streaming, Multi‑Duration CLI (v5.3)
Fixes: longitude seam and coordinate ordering so maps render correctly in Panoply.

- Re-center longitudes from 0..360 to -180..180 and sort by longitude (prevents
  disjoint blocks across the 0/360 seam; UK/IE window becomes a single slice).
- Keep latitude monotonic and tile stitching on already-sorted coords.
- apply_ufunc(np.isclose) without core dims (avoids time rechunking).
- Everything else as in v5.2: threaded Dask, WY checkpoint/resume, amax + CF time,
  return levels + GEV params, per-duration output directories.

"""

import os
import sys
import time
import argparse
import warnings
import gc
import numpy as np
import xarray as xr
import pandas as pd
from dask.distributed import Client, LocalCluster
from scipy.stats import genextreme
import cftime

warnings.filterwarnings("ignore")

# ---------------------------- CLI ----------------------------
parser = argparse.ArgumentParser(
    description="ERA5 return levels (v5.3, lon fix + sorted coords)."
)
grp = parser.add_mutually_exclusive_group(required=True)
grp.add_argument("--duration", type=int, choices=[6,12,24,48])
grp.add_argument("--durations", type=int, nargs="+", choices=[6,12,24,48])

parser.add_argument("--start-date", default="1979-10-01")
parser.add_argument("--end-date", default=None)

parser.add_argument("--lat-min", type=float, default=35.0)
parser.add_argument("--lat-max", type=float, default=65.0)
parser.add_argument("--lon-min", type=float, default=-15.0)
parser.add_argument("--lon-max", type=float, default=5.0)

parser.add_argument("--lat-block", type=int, default=40)
parser.add_argument("--lon-block", type=int, default=40)

parser.add_argument("--threads", type=int, default=4)
parser.add_argument("--mem", default="6GB")
parser.add_argument("--scratch", default="/mnt/data01/tmp")
parser.add_argument("--time-chunk", type=int, default=48)

parser.add_argument("--outdir", default=".")
parser.add_argument("--out-prefix", default="UKI")

parser.add_argument("--engine", default="netcdf4", choices=["netcdf4","h5netcdf","scipy"])
parser.add_argument("--eager-write", action="store_true")
args = parser.parse_args()

# ------------------------- Config ----------------------------
ZARR_PATH = "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"

DURATIONS = [args.duration] if args.duration else list(dict.fromkeys(args.durations))
START_DATE, END_DATE = args.start_date, args.end_date
LAT_MIN, LAT_MAX = args.lat_min, args.lat_max
LON_MIN, LON_MAX = args.lon_min, args.lon_max

LAT_BLOCK, LON_BLOCK = args.lat_block, args.lon_block
THREADS, MEM, SCRATCH, TCH = args.threads, args.mem, args.scratch, args.time_chunk
BASE_OUTDIR, PREFIX, ENGINE, EAGER = args.outdir, args.out_prefix, args.engine, args.eager_write

os.makedirs(BASE_OUTDIR, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)

# ------------------------ Helpers ----------------------------
def water_year(t):
    return xr.where(t.dt.month >= 10, t.dt.year+1, t.dt.year)

def wy_start(y): return np.datetime64(f"{y-1}-10-01")
def wy_end(y):   return np.datetime64(f"{y}-10-01")

def to_CF_time(dt64_array):
    """
    Convert an array of datetime-like values to CF time (seconds since 1970-01-01).
    Returns float64 array with np.nan for missing values.
    Fast vectorized path for numpy.datetime64 arrays.
    """
    arr = np.asarray(dt64_array)
    out = np.full(arr.shape, np.nan, dtype="float64")

    # Fast path: numpy datetime64 (any unit)
    if np.issubdtype(arr.dtype, np.datetime64):
        mask_nat = np.isnat(arr)
        if mask_nat.all():
            return out
        # convert to seconds since epoch (int64), then to float
        secs = arr.astype("datetime64[s]").astype("int64")
        out[~mask_nat] = secs[~mask_nat].astype("float64")
        return out

    # Fallback: handle mixed/object arrays elementwise
    for idx, dt in np.ndenumerate(arr):
        if dt is None:
            out[idx] = np.nan
            continue
        # pandas Timestamp
        try:
            if isinstance(dt, pd.Timestamp):
                if pd.isna(dt):
                    out[idx] = np.nan
                else:
                    out[idx] = cftime.date2num(
                        dt.to_pydatetime(),
                        units="seconds since 1970-01-01 00:00:00",
                        calendar="proleptic_gregorian"
                    )
                continue
        except Exception:
            pass
        # cftime / python datetime-like (has year)
        if hasattr(dt, "year"):
            try:
                out[idx] = cftime.date2num(
                    dt,
                    units="seconds since 1970-01-01 00:00:00",
                    calendar="proleptic_gregorian"
                )
            except Exception:
                out[idx] = np.nan
            continue
        # integers: interpret as seconds (or ns if very large)
        if isinstance(dt, (np.integer, int)):
            ival = int(dt)
            try:
                if abs(ival) > 10**12:  # heuristic: treat large ints as nanoseconds
                    py_dt = np.datetime64(ival, "ns").astype(object)
                else:
                    py_dt = np.datetime64(ival, "s").astype(object)
                out[idx] = cftime.date2num(
                    py_dt,
                    units="seconds since 1970-01-01 00:00:00",
                    calendar="proleptic_gregorian"
                )
            except Exception:
                out[idx] = np.nan
            continue
        # last resort: try coercing to numpy datetime64
        try:
            py_dt64 = np.datetime64(dt)
            if np.isnat(py_dt64):
                out[idx] = np.nan
            else:
                secs = py_dt64.astype("datetime64[s]").astype("int64")
                out[idx] = float(secs)
            continue
        except Exception:
            out[idx] = np.nan
    return out

# -------------------- Start Dask (threaded) -------------------
print(f"Starting Dask (threads={THREADS}, mem={MEM})…")
cluster = LocalCluster(
    processes=False,
    n_workers=1,
    threads_per_worker=THREADS,
    memory_limit=MEM,
    local_directory=SCRATCH,
    dashboard_address=None
)
client = Client(cluster)
client.run(lambda: __import__("dask").config.set({
    "distributed.worker.memory.target": 0.55,
    "distributed.worker.memory.spill": 0.65,
    "distributed.worker.memory.pause": 0.80,
    "distributed.worker.memory.terminate": 0.95
}))

# ------------------------ Open ERA5 ---------------------------
print("Opening ERA5 ARCO dataset…")
ds = xr.open_zarr(
    ZARR_PATH, consolidated=True, chunks={"time": TCH}, storage_options={"token": "anon"}
)

# Only keep the variables we need
ds = ds[["total_precipitation","land_sea_mask"]]

# Latitude slice (ERA5 latitude is descending; keep it monotonic)
ds = ds.sel(latitude=slice(LAT_MAX, LAT_MIN))  # e.g., 65 → 35

# v5.3: re-center longitudes to [-180, 180) and sort by longitude
# (prevents split-domain banding across 0/360 seam; UK/IE is a single continuous slice)
ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)  # re-center
ds = ds.sortby("longitude")  # ensure monotonic ascending longitudes

# Now subselect a continuous window in -180..180 (e.g., -15..5)
ds = ds.sel(longitude=slice(LON_MIN, LON_MAX))

# Land mask
ds = ds.where(ds["land_sea_mask"] > 0.5)

# Time slice
if START_DATE:
    ds = ds.sel(time=slice(START_DATE, None))
if END_DATE:
    ds = ds.sel(time=slice(None, END_DATE))

tp = ds["total_precipitation"]
if tp.sizes["time"] == 0:
    print("No time samples after slicing.")
    sys.exit(2)

# ------------------ Determine complete WYs --------------------
wy_all = water_year(tp.time).astype("int32").compute().values
years_all = np.unique(wy_all)
t0, tN = tp.time.values[0], tp.time.values[-1]
years = [y for y in years_all if (wy_start(y) >= t0 and wy_end(y) <= tN)]
if not years:
    print("No complete water years.")
    sys.exit(2)
print(f"Water years: {years[0]}–{years[-1]}")

# ---------------- MAIN LOOP over durations -------------------
for DURATION in DURATIONS:
    print("\n" + "="*72)
    print(f"Starting duration {DURATION}h (v5.3 lon-order fix)")
    print("="*72)

    OUTDIR = os.path.join(BASE_OUTDIR, f"{DURATION}h")
    os.makedirs(OUTDIR, exist_ok=True)

    CKDIR = os.path.join(OUTDIR, "checkpoints")
    os.makedirs(CKDIR, exist_ok=True)

    done_wy = {
        int(f.split("WY")[1].split(".")[0])
        for f in os.listdir(CKDIR)
        if f.startswith("checkpoint_WY") and f.endswith(".nc")
    }

    # Rolling sum lazily
    rolling = tp.rolling(time=DURATION, min_periods=DURATION).sum().astype("float32")
    lat_sz = rolling.sizes["latitude"]
    lon_sz = rolling.sizes["longitude"]

    # -------------- per-WY streaming + tiling -----------------
    for y in years:
        if y in done_wy:
            print(f"Skipping WY {y} (checkpoint exists)")
            continue

        print(f"Processing WY {y}…")
        block_y = rolling.sel(time=slice(wy_start(y), wy_end(y)))
        if block_y.sizes["time"] == 0:
            print(" empty → skip")
            continue

        row_vals_all = []
        row_times_all = []

        # latitude rows
        for i0 in range(0, lat_sz, LAT_BLOCK):
            i1 = min(i0 + LAT_BLOCK, lat_sz)
            row_vals = []
            row_times = []

            for j0 in range(0, lon_sz, LON_BLOCK):
                j1 = min(j0 + LON_BLOCK, lon_sz)

                # Extract tile (coordinates are already sorted and continuous)
                block = block_y.isel(latitude=slice(i0,i1), longitude=slice(j0,j1))

                # annual max per pixel in this tile
                maxvals = block.max("time")

                # equality test (latest occurrence): isclose → reverse → argmax
                isclose = xr.apply_ufunc(
                    np.isclose, block, maxvals,
                    dask="parallelized", output_dtypes=[bool],
                    kwargs={"rtol":1e-6, "atol":1e-12},
                )
                isrev = isclose.isel(time=slice(None, None, -1))
                idx_rev = isrev.argmax("time")
                T = block.sizes["time"]
                idx = (T-1) - idx_rev
                idx = idx.compute()

                ts = block["time"].isel(time=idx)

                row_vals.append(maxvals.rename("amax_value"))
                row_times.append(ts.rename("amax_datetime"))

            # concat tiles across longitudes (already monotonic), then store row
            row_vals_all.append(xr.concat(row_vals, dim="longitude"))
            row_times_all.append(xr.concat(row_times, dim="longitude"))

        # concat latitude rows (ERA5 latitude is descending; keep monotonic order)
        year_vals = xr.concat(row_vals_all, dim="latitude")
        year_time = xr.concat(row_times_all, dim="latitude")

        # sanity: ensure coordinates are strictly monotonic and dims ordered
        year_vals = year_vals.sortby(["latitude", "longitude"])
        year_time = year_time.sortby(["latitude", "longitude"])
        year_vals = year_vals.transpose("latitude","longitude")
        year_time = year_time.transpose("latitude","longitude")

        ck = xr.Dataset({
            "amax_value":   year_vals.expand_dims(water_year=[y]),
            "amax_datetime":year_time.expand_dims(water_year=[y]),
        })

        ckfile = os.path.join(CKDIR, f"checkpoint_WY{y}.nc")
        print(f"Writing checkpoint {ckfile}")
        if EAGER:
            ck = ck.compute()
        ck.to_netcdf(ckfile, engine=ENGINE)
        gc.collect()
        print(f" ✓ WY {y} done")

    # -------------- assemble checkpoints ----------------------
    print("Assembling checkpoints…")
    ckfiles = sorted([
        os.path.join(CKDIR, f)
        for f in os.listdir(CKDIR)
        if f.startswith("checkpoint_WY")
    ])

    vals, dts = [], []
    for fn in ckfiles:
        dsck = xr.open_dataset(fn)
        vals.append(dsck["amax_value"])
        dts.append(dsck["amax_datetime"])

    amax_value = xr.concat(vals, dim="water_year").sortby(["latitude","longitude"])
    amax_datetime = xr.concat(dts, dim="water_year").sortby(["latitude","longitude"])

    print("Creating CF‑compliant amax_time_CF…")
    cf = xr.DataArray(
        to_CF_time(amax_datetime.values),
        dims=amax_datetime.dims,
        coords=amax_datetime.coords,
        name="amax_time_CF"
    )
    cf.attrs.update({
        "units":"seconds since 1970-01-01 00:00:00",
        "calendar":"proleptic_gregorian"
    })

    # --------------------- Fit GEV ----------------------------
    print("Fitting GEV…")
    def fit_gev(arr):
        arr = np.asarray(arr)
        arr = arr[~np.isnan(arr)]
        if arr.size < 20:
            return np.nan, np.nan, np.nan
        c,loc,scale = genextreme.fit(arr)
        return c,loc,scale

    gevp = xr.apply_ufunc(
        fit_gev, amax_value,
        input_core_dims=[["water_year"]],
        output_core_dims=[[],[],[]],
        vectorize=True, dask="parallelized",
        output_dtypes=[float,float,float]
    )
    shape = gevp[0].rename("shape")
    loc   = gevp[1].rename("loc")
    scale = gevp[2].rename("scale")

    # ---------------- Return levels ---------------------------
    RPs = np.array([2,5,10,20,30,50,75,100,200,500,1000,1500], dtype=float)
    def rl_fn(c,loc,scale,rp):
        q = 1 - 1/np.asarray(rp)
        return np.array([genextreme.ppf(qq, c,loc,scale) for qq in q],
                        dtype=np.float32)

    rl = xr.apply_ufunc(
        rl_fn, shape, loc, scale,
        kwargs={"rp":RPs},
        input_core_dims=[[],[],[]],
        output_core_dims=[["rl"]],
        vectorize=True, dask="parallelized",
        output_dtypes=[np.float32]
    ).assign_coords(rl=RPs).rename("return_level")

    wy0, wyN = years[0], years[-1]
    file_return = os.path.join(OUTDIR, f"{PREFIX}_return_levels_{DURATION}h_WY{wy0}-{wyN}.nc")
    file_amax   = os.path.join(OUTDIR, f"{PREFIX}_amax_{DURATION}h_WY{wy0}-{wyN}.nc")

    print(f"Writing return levels → {file_return}")
    outA = xr.Dataset({"return_level": rl, "shape": shape, "loc": loc, "scale": scale})
    if EAGER:
        outA = outA.compute()
    outA.to_netcdf(file_return, engine=ENGINE)

    print(f"Writing annual maxima → {file_amax}")
    outB = xr.Dataset({"amax_value": amax_value, "amax_datetime": amax_datetime, "amax_time_CF": cf})
    if EAGER:
        outB = outB.compute()
    outB.to_netcdf(file_amax, engine=ENGINE)

    print(f" ✓ Completed {DURATION}h")

print("All durations finished.")
try:
    client.close(); cluster.close()
except:
    pass