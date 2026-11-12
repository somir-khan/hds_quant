"""Combine explicit coverage jobs and check standalone timing/quality assignment parity."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds_quant.coverage_summary import aggregate

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job_dirs', nargs='+'); p.add_argument('--output-dir', required=True)
    a = p.parse_args(); print(aggregate(a.job_dirs, a.output_dir))
