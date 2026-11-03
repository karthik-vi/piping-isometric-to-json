"""Evaluation: field exact-match, drawing-level exact match, and set F1."""
from __future__ import annotations

import json
from pathlib import Path

from .pipeline import SCALAR_FIELDS

SET_FIELDS = ["tags", "dimensions_mm", "bom"]


def _as_set(field: str, v) -> set:
    if field == "bom":
        return {(r["item"], r["qty"], r["unit"], r["description"]) for r in (v or [])}
    return set(v or [])


def prf(gold: set, pred: set) -> tuple[float, float, float, int, int, int]:
    tp = len(gold & pred)
    p = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    r = tp / len(gold) if gold else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp, len(pred), len(gold)


def score_one(gt: dict, pred: dict) -> dict:
    out = {"fields": {}}
    for f in SCALAR_FIELDS:
        out["fields"][f] = gt[f] == pred.get(f)
    for f in SET_FIELDS:
        out["fields"][f] = _as_set(f, gt[f]) == _as_set(f, pred.get(f))
        out[f + "_counts"] = prf(_as_set(f, gt[f]), _as_set(f, pred.get(f)))[3:]
    out["drawing_exact"] = all(out["fields"].values())
    out["scalar_exact"] = all(out["fields"][f] for f in SCALAR_FIELDS)
    meta = pred.get("_meta", {})
    out["flagged"] = bool(meta.get("needs_review"))
    out["latency_s"] = meta.get("latency_s")
    return out


def aggregate(scores: list[dict]) -> dict:
    n = len(scores)
    res = {"n": n}
    allf = SCALAR_FIELDS + SET_FIELDS
    res["field_exact_match"] = {f: sum(s["fields"][f] for s in scores) / n for f in allf}
    res["scalar_field_exact_match_micro"] = sum(s["fields"][f] for s in scores for f in SCALAR_FIELDS) / (n * len(SCALAR_FIELDS))
    res["drawing_exact_match"] = sum(s["drawing_exact"] for s in scores) / n
    res["scalar_block_exact_match"] = sum(s["scalar_exact"] for s in scores) / n
    for f in SET_FIELDS:
        tp = sum(s[f + "_counts"][0] for s in scores)
        npred = sum(s[f + "_counts"][1] for s in scores)
        ngold = sum(s[f + "_counts"][2] for s in scores)
        p = tp / npred if npred else 0.0
        r = tp / ngold if ngold else 0.0
        res[f + "_f1"] = 2 * p * r / (p + r) if p + r else 0.0
        res[f + "_precision"], res[f + "_recall"] = p, r
    # how well does the validator catch errors? (flag = needs_review non-empty)
    wrong = [s for s in scores if not s["drawing_exact"]]
    res["error_drawings_flagged_rate"] = sum(s["flagged"] for s in wrong) / len(wrong) if wrong else None
    res["correct_drawings_flagged_rate"] = (sum(s["flagged"] for s in scores if s["drawing_exact"]) /
                                            max(1, sum(s["drawing_exact"] for s in scores)))
    lat = sorted(s["latency_s"] for s in scores if s["latency_s"] is not None)
    res["latency_s_median"] = lat[len(lat) // 2] if lat else None
    return res


def load_gt(png: Path) -> dict:
    return json.loads(png.with_suffix(".json").read_text())
