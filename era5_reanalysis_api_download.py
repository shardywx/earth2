#!/usr/bin/env python3
import os
import time
import logging
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import cdsapi
import calendar
import numpy as np

# logging
LOGFILE = "glofas_download.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    handlers=[logging.FileHandler(LOGFILE),
                              logging.StreamHandler()])

def read_cdsapirc(path):
    """ 
    Read data from a specified URL using the CDS API 
    """
    d = {}
    with open(path) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                d[k.strip()] = v.strip()
    return d
cfg = read_cdsapirc('/home/jbanorthwest.co.uk/samhardy/.cdsapirc_cds')

# configuration: the specific years/months you requested
FULL_YEARS = np.arange(1980,2025) # years with all months

DATASET = "derived-era5-single-levels-daily-statistics"
BBOX = [60, -15, 45, 10]  # N W S E

BASE_REQUEST = {
    "variable": ["total_precipitation"],
    "daily_statistic": ["daily_maximum"], # daily_maximum
    "product_type": ["reanalysis"],
    "time_zone": ["utc+00:00"],
    "frequency": ["1_hourly"], # "1_hourly" for "daily_sum"
    "area": BBOX,
    "format": "grib2",
}

MAX_WORKERS = 10
MAX_ATTEMPTS = 5
SLEEP_BETWEEN_SUBMITS = 1

def worker_task(year, month, days, target_dir="/mnt/metdata/W25-2348/era5_daily_stats/1h_max/"):
    client = cdsapi.Client(url=cfg['url'], key=cfg['key'])
    req = copy.deepcopy(BASE_REQUEST)
    req.update({"year": [str(year)], "month": [month], "day": days})
    target_fname = os.path.join(target_dir, f"{DATASET}_{year}_month{month}.grib2")
    if os.path.exists(target_fname) and os.path.getsize(target_fname) > 0:
        logging.info("Skipping existing file %s", target_fname)
        return target_fname
    start = time.time()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logging.info("Requesting year=%s month=%s days=%s attempt %d", year, month, ",".join(days), attempt)
            client.retrieve(DATASET, req, target_fname)
            elapsed = time.time() - start
            logging.info("Finished %s in %.1f s", target_fname, elapsed)
            return target_fname
        except Exception as e:
            logging.warning("Year=%s month=%s attempt %d failed: %s", year, month, attempt, e)
            if attempt == MAX_ATTEMPTS:
                logging.error("Giving up on year=%s month=%s after %d attempts", year, month, MAX_ATTEMPTS)
                raise
            backoff = 10 * (2 ** (attempt - 1))
            logging.info("Sleeping for %d s before retry", backoff)
            time.sleep(backoff)
    return None

def main():
    tasks = []
    # add all months for each FULL_YEARS
    for y in FULL_YEARS:
        for m in range(1, 13):
            ndays = calendar.monthrange(y, m)[1]
            days = [f"{d:02d}" for d in range(1, ndays+1)] 
            month = f"{m:02d}"
            tasks.append((y, month, days))

    logging.info("Submitting %d tasks (total)", len(tasks))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {}
        for year, month, days in tasks:
            futures[exe.submit(worker_task, year, month, days)] = (year, month)
            time.sleep(SLEEP_BETWEEN_SUBMITS)
        for fut in as_completed(futures):
            year, month = futures[fut]
            try:
                res = fut.result()
                logging.info("Task for %s-%s completed: %s", year, month, res)
            except Exception as e:
                logging.error("Task for %s-%s failed: %s", year, month, e)

if __name__ == "__main__":
    main()