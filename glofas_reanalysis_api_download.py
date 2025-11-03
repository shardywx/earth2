#!/usr/bin/env python3
import os
import time
import logging
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import cdsapi
import calendar

# logging
LOGFILE = "glofas_download.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    handlers=[logging.FileHandler(LOGFILE),
                              logging.StreamHandler()])

# configuration: the specific years/months you requested
FULL_YEARS = [2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023] # years with all months

DATASET = "cems-glofas-historical"
BBOX = [55, -8, 52, -5]  # N W S E

BASE_REQUEST = {
    "system_version": ["version_4_0"],
    "variable": ["river_discharge_in_the_last_24_hours"],
    "hydrological_model": ["lisflood"],
    "product_type": ["consolidated"], # "product_type": ["control_reforecast"],
    "area": BBOX,
    "data_format": "grib2",
    "download_format": "zip",
}

MAX_WORKERS = 10
MAX_ATTEMPTS = 5
SLEEP_BETWEEN_SUBMITS = 1

def worker_task(year, month, days, target_dir="/mnt/metdata/W25-2348/glofas/"):
    client = cdsapi.Client()
    req = copy.deepcopy(BASE_REQUEST)
    req.update({"hyear": [str(year)], "hmonth": [month], "hday": days})
    target_fname = os.path.join(target_dir, f"{DATASET}_{year}_hmonth{month}.zip")
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