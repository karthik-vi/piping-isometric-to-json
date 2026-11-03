"""End-to-end: drawing.png -> validated.json

  python scripts/run_pipeline.py data/samples/iso_0000.png -o out/iso_0000.json [--vlm smolvlm|anthropic|none]
Exit code 0 if the output passes validation, 2 if fields need human review.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iso2json.pipeline import PipelineConfig, run_file  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("image", type=Path)
ap.add_argument("-o", "--out", type=Path)
ap.add_argument("--vlm", default=None, help="none | anthropic | smolvlm (default: $VLM_PROVIDER or none)")
a = ap.parse_args()
import os  # noqa: E402

cfg = PipelineConfig(vlm_provider=a.vlm or os.getenv("VLM_PROVIDER", "none"))
rec = run_file(a.image, cfg)
txt = json.dumps(rec, indent=2)
if a.out:
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(txt)
    print(f"wrote {a.out}  valid={rec['_meta']['valid']}  needs_review={rec['_meta']['needs_review']}")
else:
    print(txt)
sys.exit(0 if rec["_meta"]["valid"] else 2)
