"""Generate a synthetic isometric dataset (PNG + ground-truth JSON).

  python scripts/generate_dataset.py --out data/synthetic/test --n 60 --seed 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iso2json.generator import write_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--scanned-frac", type=float, default=0.5)
a = ap.parse_args()
paths = write_dataset(a.out, a.n, a.seed, a.scanned_frac)
print(f"wrote {len(paths)} drawings to {a.out}")
