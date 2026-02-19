#!/bin/bash
# Example runner script for 2 durations
# NOTE --scratch needs to be a locally mounted disk preferably an SSD - the name is vm dependedent:
# ski-wrf03-vm /mnt/local/tmp
# ski-tcrm01-vm /mnt/data01/tmp

python -u run_return_levels_cli_v5_sh.py \
  --durations 48 \
  --start-date 1979-09-01 \
  --threads 6 \
  --mem 12GB \
  --lat-block 60 --lon-block 60 \
  --time-chunk 48 \
  --engine netcdf4 \
  --outdir /mnt/metdata/2025s1985/era5_eva/ \
  --scratch /mnt/data01/tmp