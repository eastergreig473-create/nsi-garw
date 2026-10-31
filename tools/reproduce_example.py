"""Verify the genuine, fixed-fit HPatches SIFT compact result."""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pipeline/src'))
spec = importlib.util.spec_from_file_location('compact_recompute', ROOT / 'pipeline/scripts/recompute_compact_nsi.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--groups', action='store_true')
args = parser.parse_args()
module.main(ROOT / 'examples/hpatches_sift_compact.npz', args.groups)
