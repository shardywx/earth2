#!/usr/bin/env python3
import os
import time
import logging
import copy
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
import cdsapi

# logging
LOGFILE = "glofas_download.log"
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    handlers=[logging.FileHandler(LOGFILE),
                              logging.StreamHandler()])

# # config
# DATASET = "cems-glofas-reforecast"
# BBOX = [55, -8, 52, -5]  # N W S E
# YEARS = ["%d" % (y) for y in range(2011, 2017)]
# LEADTIMES = ["%d" % (l) for l in range(24, 240, 24)]
# MAX_WORKERS = 4         # change to 2-4; avoid high concurrency
# MAX_ATTEMPTS = 5
# SLEEP_BETWEEN_TASKS = 2  # seconds

# config
START_YEAR = 2014
END_YEAR = 2019

DATASET = "cems-glofas-reforecast"
BBOX = [55, -8, 52, -5]  # N W S E
YEARS = [str(y) for y in range(START_YEAR, END_YEAR + 1)]
LEADTIMES = [str(l) for l in range(24, 240, 24)]

MONTH_DAYS = {
    '01': ["01","04","08","11","15","18","22","25","29"],
    '02': ["01","05","08","12","15","19","22","26","28"],
    '03': ["04","07","11","14","18","21","25","27","28","30"],
    '04': ["01","03","04","06","08","10","11","13","15","17","18","20","22","24","25","27","29"],
    '05': ["01","02","04","06","08","09","11","13","15","16","18","20","22","23","25","27","29","30"],
    '06': ["01","03","05","06","08","10","12","13","15","17","19","20","22","24","26","27","29"],
    '07': ["01","03","04","06","08","10","11","13","15","17","18","20","22","24","25","27","29","31"],
    '08': ["01","03","05","07","08","10","12","14","15","17","19","21","22","24","26","28","29","31"],
    '09': ["02","04","05","07","09","11","12","14","16","18","19","21","23","25","26","28","30"],
    '10': ["02","03","05","07","09","10","12","14","16","17","19","21","23","24","26","28","30","31"],
    '11': ["02","04","06","07","09","11","13","14","16","18","20","21","23","25","27","30"],
    '12': ["04","07","11","14","18","21","25","28"]
}

BASE_REQUEST = {
    "system_version": ["version_4_0"],
    "variable": ["river_discharge_in_the_last_24_hours"],
    "hydrological_model": ["lisflood"],
    "product_type": ["ensemble_perturbed_reforecast"],
    #"hyear": YEARS, # will be set per task
    "area": BBOX,
    "leadtime_hour": LEADTIMES,
    "data_format": "grib2",
    "download_format": "zip",
}

MAX_WORKERS = 4       # recommended 2-4
MAX_ATTEMPTS = 5
SLEEP_BETWEEN_SUBMITS = 1  # small pause to avoid flooding queue

# # graceful shutdown flag
# STOP = False
# def _signal_handler(signum, frame):
#     global STOP
#     STOP = True
#     logging.warning("Received signal %s, will stop submitting new tasks and exit when running tasks finish.", signum)
# signal.signal(signal.SIGINT, _signal_handler)
# signal.signal(signal.SIGTERM, _signal_handler)


def worker_task(year, month, days, target_dir="."):
    # each worker creates its own client
    client = cdsapi.Client()
    # request hmonth as single-item list (API supports list)
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
            # use the 3-arg retrieve form which writes directly to target
            client.retrieve(DATASET, req, target_fname)
            elapsed = time.time() - start
            logging.info("Finished %s in %.1f s", target_fname, elapsed)
            return target_fname
        except Exception as e:
            logging.warning("Year=%s month %s attempt %d failed: %s", year, month, attempt, e)
            if attempt == MAX_ATTEMPTS:
                logging.error("Giving up on year=%s month %s after %d attempts", year, month, MAX_ATTEMPTS)
                raise
            backoff = 10 * (2 ** (attempt - 1))
            logging.info("Sleeping for %d s before retry", backoff)
            time.sleep(backoff)
    return None


def main():
    # months = list(MONTH_DAYS.items())  # list of (month, days)
    # with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
    #     futures = {}
    #     for year, month, days in months:
    #         if STOP:
    #             logging.info("Stop requested; not submitting more tasks.")
    #             break
    #         futures[exe.submit(worker_task, month, days)] = month
    #         time.sleep(SLEEP_BETWEEN_TASKS)  # small throttle when submitting

    tasks = []
    for y in range(START_YEAR, END_YEAR + 1):
        for month, days in MONTH_DAYS.items():
            tasks.append((y, month, days))

    logging.info("Submitting %d tasks (years %d-%d, months=%d)", len(tasks), START_YEAR, END_YEAR, len(MONTH_DAYS))

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {}
        for year, month, days in tasks:
            futures[exe.submit(worker_task, year, month, days)] = (year, month)
            time.sleep(SLEEP_BETWEEN_SUBMITS)

        # collect results
        for fut in as_completed(futures):
            year, month = futures[fut]
            try:
                res = fut.result()
                logging.info("Task for %s-%s completed: %s", year, month, res)
            except Exception as e:
                logging.error("Task for %s-%s failed: %s", year, month, e)

if __name__ == "__main__":
    main()