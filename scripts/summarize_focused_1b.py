"""Rebuild the verified summary of one completed combined 1B run."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hds_quant.focused_1b_summary import aggregate

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('run_root')
    a = p.parse_args(); print(aggregate(a.run_root))
