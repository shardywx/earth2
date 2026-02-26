#!/usr/bin/env python3
import zipfile
from pathlib import Path


def main():
    for p in Path('/mnt/metdata/W25-2348/glofas/').glob('*.zip'):
        dest = p.with_suffix('') # folder named like file (no .zip)
        dest.mkdir(exist_ok=True)
        with zipfile.ZipFile(p, 'r') as z:
            z.extractall(dest)

if __name__ == "__main__":
    main()