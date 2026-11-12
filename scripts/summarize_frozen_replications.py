"""Aggregate explicit completed job directories; reject duplicate seeds and old diagnostics."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds_quant.frozen_replication_summary import aggregate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job_dirs', nargs='+')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    for row in aggregate(args.job_dirs, args.output_dir):
        print(row)
