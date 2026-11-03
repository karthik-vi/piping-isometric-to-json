"""Run pipeline variants (ablations) over a labelled folder and report metrics.

  python scripts/evaluate.py --data data/synthetic/test --variants A,B,C,D,E --out results/test.json
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iso2json.evaluate import aggregate, load_gt, score_one  # noqa: E402
from iso2json.pipeline import PipelineConfig, run_file  # noqa: E402

ROT = (0, 30, -30, 90, -90)
VARIANTS = {
    "A": ("full-page OCR + regex (baseline)", dict(use_layout=False, rotations=(0,), use_repair=False, preprocess=False, table_rows=False, dim_filter=False, tag_rules=False, dual_read=False)),
    "B": ("+ layout crops", dict(rotations=(0,), use_repair=False, preprocess=False, table_rows=False, dim_filter=False, tag_rules=False, dual_read=False)),
    "C": ("+ deskew/denoise/upscale + row-wise table OCR", dict(rotations=(0,), use_repair=False, dim_filter=False, tag_rules=False, dual_read=False)),
    "D": ("+ multi-rotation drawing OCR + dim filter", dict(use_repair=False, tag_rules=False, dual_read=False)),
    "E": ("+ cross-field validation/repair (full rules)", dict(tag_rules=False, dual_read=False)),
    "E2": ("+ tag rules (fuzzy prefix, fragment merge, dual read)", dict()),
    "F": ("+ VLM re-read of flagged fields (SmolVLM, local)", dict(vlm_provider="smolvlm")),
    "G": ("+ VLM re-read of flagged fields (Anthropic API)", dict(vlm_provider="anthropic")),
}


def _run(args):
    png, kw = args
    pred = run_file(png, PipelineConfig(**kw))
    return png.name, score_one(load_gt(png), pred), pred


ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--variants", default="A,B,C,D,E")
ap.add_argument("--out", type=Path)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

pngs = sorted(a.data.glob("*.png"))
if a.limit:
    pngs = pngs[: a.limit]
report = {"data": str(a.data), "n": len(pngs), "variants": {}}
for v in a.variants.split(","):
    desc, kw = VARIANTS[v]
    t0 = time.time()
    workers = 1 if kw.get("vlm_provider") == "smolvlm" else a.workers
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_run, [(p, kw) for p in pngs]))
    scores = [r[1] for r in rows]
    res = aggregate(scores)
    for split, tag in (("clean", lambda n: "_scan" not in n), ("scanned", lambda n: "_scan" in n)):
        sub = [s for (name, s, _) in rows if tag(name)]
        if sub:
            res[split] = {k: aggregate(sub)[k] for k in ("n", "drawing_exact_match", "scalar_field_exact_match_micro", "tags_f1", "dimensions_mm_f1", "bom_f1")}
    res["failures"] = {name: [f for f, ok in s["fields"].items() if not ok] for name, s, _ in rows if not s["drawing_exact"]}
    res["wall_s"] = round(time.time() - t0, 1)
    report["variants"][v] = {"desc": desc, "config": kw, **res}
    print(f"{v} {desc:55s} drawEM={res['drawing_exact_match']:.3f} fieldEM={res['scalar_field_exact_match_micro']:.3f} "
          f"tagsF1={res['tags_f1']:.3f} dimsF1={res['dimensions_mm_f1']:.3f} bomF1={res['bom_f1']:.3f} "
          f"errFlagged={res['error_drawings_flagged_rate']} wall={res['wall_s']}s", flush=True)
if a.out:
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, default=str))
