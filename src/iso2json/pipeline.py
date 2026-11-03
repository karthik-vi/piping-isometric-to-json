"""End-to-end pipeline: drawing.png -> validated JSON.

Stages (each can be toggled for ablations):
  1. layout   - find drawing frame / BOM / title block by ruled-line detection
  2. ocr      - Tesseract on each region; drawing area at several rotations
  3. rules    - regex/grammar parsing + OCR-confusion fixes per field
  4. repair   - cross-field validation and repair (line no <-> size/spec/service,
                spec -> material via piping class table, BOM numbering)
  5. vlm      - optional Vision-Language Model re-read of fields that still
                fail validation (provider: none | anthropic | smolvlm)
"""
from __future__ import annotations

import difflib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema
from PIL import Image

from . import config as C
from . import ocr, parse

SCHEMA = json.loads((Path(__file__).resolve().parents[2] / "schemas" / "isometric.schema.json").read_text())
SCALAR_FIELDS = ["drawing_number", "line_number", "nominal_size_in", "service", "pipe_spec", "material",
                 "design_pressure_barg", "design_temp_c", "revision"]


@dataclass
class PipelineConfig:
    use_layout: bool = True
    rotations: tuple[int, ...] = (0, 30, -30, 90, -90)
    use_repair: bool = True
    vlm_provider: str = "none"  # none | anthropic | smolvlm
    min_dim_conf: float = 40.0
    tag_rules: bool = True  # tags: 0deg only, fuzzy prefix, merge split words, location dedupe
    dual_read: bool = True  # extra 0deg OCR read on the raw (non-binarized) drawing
    dim_filter: bool = True  # vote/substring filter for dimension candidates (part of rules stage)
    min_dim_votes: int = 2
    dim_conf_hi: float = 85.0
    table_rows: bool = True  # OCR tables row-by-row between ruling lines
    preprocess: bool = True  # denoise + 2x upscale + Otsu + ruled-line removal
    extra: dict = field(default_factory=dict)


def _template_regions(w: int, h: int):
    """Fallback template coordinates (fractions of sheet) if ruled-box detection fails."""
    sx, sy = w / 2000, h / 1400
    s = lambda b: (int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy))
    return s((40, 40, 1300, 1360)), s((1340, 40, 1960, 700)), s((1340, 960, 1960, 1360))


def find_regions(gray):
    h, w = gray.shape
    boxes = ocr.detect_boxes(gray)
    left = [b for b in boxes if b[0] < 0.3 * w and (b[2] - b[0]) > 0.4 * w]
    right = sorted([b for b in boxes if b[0] > 0.55 * w], key=lambda b: b[1])
    if left and len(right) >= 2:
        return (min(left, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])), right[0], right[-1]), "detected"
    return _template_regions(w, h), "template_fallback"


def extract(img: Image.Image, cfg: PipelineConfig | None = None) -> dict:
    cfg = cfg or PipelineConfig()
    t0 = time.perf_counter()
    timeouts0 = ocr.TIMEOUTS["count"]
    gray = ocr.to_gray(img)
    meta: dict = {"stages": [], "provenance": {}, "needs_review": []}
    if cfg.preprocess:
        gray, ang = ocr.deskew(gray)
        meta["deskew_deg"] = round(ang, 2)

    # 1-2. layout + OCR
    if cfg.use_layout:
        (draw_box, bom_box, title_box), how = find_regions(gray)
        meta["layout"] = how
        tc, bc, draw = ocr.crop(gray, title_box), ocr.crop(gray, bom_box), ocr.crop(gray, draw_box, pad=8)
        if cfg.preprocess:
            tc, bc, draw = ocr.prep(tc, remove_rules=True), ocr.prep(bc, remove_rules=True), ocr.prep(draw)
        if cfg.table_rows:
            title_lines = ocr.table_rows(ocr.crop(gray, title_box), preprocess=cfg.preprocess)
            bom_lines = ocr.table_rows(ocr.crop(gray, bom_box), preprocess=cfg.preprocess)
        else:
            title_lines = ocr.ocr_lines(tc, psm=6)
            bom_lines = ocr.ocr_lines(bc, psm=6)
    else:  # baseline: one sparse-text pass on the full sheet
        meta["layout"] = "none"
        title_lines = bom_lines = ocr.ocr_lines(gray, psm=11)
        draw = gray
    meta["stages"].append("ocr")

    words = []
    for a in cfg.rotations:
        g = draw if a == 0 else ocr.rotate(draw, a)
        words += ocr.ocr_words(g, psm=11, angle=a)
    zero_words = [w for w in words if w.angle == 0]
    if cfg.dual_read and cfg.use_layout and cfg.preprocess:
        raw = ocr.crop(gray, draw_box, pad=8)
        for w in ocr.ocr_words(raw, psm=11, angle=0):  # map 1x coords onto the 2x preprocessed frame
            w.x, w.y, w.w, w.h = w.x * 2, w.y * 2, w.w * 2, w.h * 2
            zero_words.append(w)

    # 3. rules
    rec = parse.parse_title_block(title_lines)
    bom = parse.parse_bom(bom_lines)
    tags, dims, line_votes = Counter(), Counter(), Counter()
    dim_conf: dict[int, float] = {}
    if cfg.tag_rules:
        tags = _tag_candidates(zero_words)
    for wd in words:
        if (t := parse.parse_tag(wd.text, fuzzy_prefix=cfg.tag_rules)):
            if not cfg.tag_rules:
                tags[t] += 1
        elif (d := parse.parse_dim(wd.text)) and wd.conf >= cfg.min_dim_conf:
            dims[d] += 1
            dim_conf[d] = max(dim_conf.get(d, 0.0), wd.conf)
        elif (ln := parse.fix_line_number(wd.text)):
            line_votes[ln] += 1
    for f in SCALAR_FIELDS:
        rec.setdefault(f, None)
        meta["provenance"][f] = "ocr" if rec[f] is not None else "missing"
    rec["service"] = None
    meta["provenance"]["service"] = "missing"
    if rec.get("line_number") and C.LINE_NO_RE.fullmatch(str(rec["line_number"])):
        rec["service"] = rec["line_number"].split("-")[1]
        meta["provenance"]["service"] = "rule:from_line_number"
    rec["bom"] = bom
    rec["tags"] = sorted(tags)
    if cfg.dim_filter:
        dims = _filter_dims(dims, dim_conf, tags, cfg)
    rec["dimensions_mm"] = sorted(dims)
    meta["stages"].append("rules")

    # 4. repair
    if cfg.use_repair:
        _repair(rec, line_votes, meta)
        meta["stages"].append("repair")

    # validation
    problems = validate(rec)
    # 5. VLM (only for fields still failing)
    if problems and cfg.vlm_provider != "none" and cfg.use_layout:
        from .vlm import refine_fields
        refine_fields(rec, problems, ocr.crop(gray, title_box), cfg.vlm_provider, meta)
        if cfg.use_repair:  # propagate VLM-fixed fields (e.g. line no -> service/size/spec)
            _repair(rec, Counter(), meta)
        problems = validate(rec)
        meta["stages"].append(f"vlm:{cfg.vlm_provider}")
    meta["needs_review"] = sorted(problems)
    meta["valid"] = not problems
    meta["latency_s"] = round(time.perf_counter() - t0, 3)
    meta["ocr_timeouts"] = ocr.TIMEOUTS["count"] - timeouts0
    rec["_meta"] = meta
    return rec


def _tag_candidates(words) -> Counter:
    """Tag reads from horizontal passes. Adjacent fragments on one line are merged
    ('PSV' '-' '5988'); overlapping reads of the same label keep the most confident."""
    cands = []  # (conf, box, tag)
    by_line: dict = {}
    for w in words:
        by_line.setdefault((id(w.line_key), w.line_key, w.h > 0), []).append(w)
    for w in words:
        if (t := parse.parse_tag(w.text)):
            cands.append((w.conf, (w.x, w.y, w.x + w.w, w.y + w.h), t))
    # merge fragments within the same tesseract line
    lines: dict = {}
    for w in words:
        lines.setdefault((w.line_key, w.y // 40), []).append(w)
    for ws in lines.values():
        ws = sorted(ws, key=lambda w: w.x)
        for i in range(len(ws)):
            for j in (i + 1, i + 2):
                if j >= len(ws):
                    break
                seg = ws[i: j + 1]
                if seg[-1].x - (seg[0].x + seg[0].w) > 6 * max(w.h for w in seg):
                    break
                t = parse.parse_tag("".join(w.text for w in seg))
                if t:
                    x0, y0 = min(w.x for w in seg), min(w.y for w in seg)
                    x1, y1 = max(w.x + w.w for w in seg), max(w.y + w.h for w in seg)
                    cands.append((min(w.conf for w in seg), (x0, y0, x1, y1), t))
    kept: list = []
    for c in sorted(cands, key=lambda c: -c[0]):
        if not any(_iou(c[1], k[1]) > 0.2 for k in kept):
            kept.append(c)
    return Counter(k[2] for k in kept)


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0.0


def _filter_dims(dims: Counter, conf: dict, tags: Counter, cfg: PipelineConfig) -> Counter:
    """Keep dimension reads that are corroborated (multiple rotation passes or high confidence),
    and drop fragments of longer numbers (e.g. '550' cut from '4550') or of tag numbers."""
    tag_nums = {t.split("-")[1] for t in tags}
    keep = Counter({d: v for d, v in dims.items()
                    if (v >= cfg.min_dim_votes or conf.get(d, 0) >= cfg.dim_conf_hi) and str(d) not in tag_nums})
    out = Counter()
    for d, v in keep.items():
        sd = str(d)
        if any(sd != str(o) and sd in str(o) and keep[o] >= v for o in keep):
            continue
        out[d] = v
    return out


def _repair(rec: dict, line_votes: Counter, meta: dict) -> None:
    prov = meta["provenance"]
    ln = rec.get("line_number")
    ln_ok = bool(ln and C.LINE_NO_RE.fullmatch(str(ln)))
    if not ln_ok and line_votes:  # title block unreadable: use the label on the pipe
        rec["line_number"], ln_ok = line_votes.most_common(1)[0][0], True
        prov["line_number"] = "rule:drawing_label"
    if ln_ok:
        size, svc, _, spec = rec["line_number"].split("-")
        if rec.get("nominal_size_in") not in C.NPS_SIZES or rec.get("nominal_size_in") != int(size):
            if int(size) in C.NPS_SIZES:
                rec["nominal_size_in"] = int(size)
                prov["nominal_size_in"] = "rule:from_line_number"
        if rec.get("pipe_spec") not in C.PIPING_CLASS_TABLE and spec in C.PIPING_CLASS_TABLE:
            rec["pipe_spec"] = spec
            prov["pipe_spec"] = "rule:from_line_number"
        if svc in C.SERVICE_CODES:
            rec["service"] = svc
    spec = rec.get("pipe_spec")
    if spec in C.PIPING_CLASS_TABLE and rec.get("material") != C.PIPING_CLASS_TABLE[spec]:
        mat = rec.get("material") or ""
        # Snap only when OCR text is plausibly the table value (avoid silently overwriting a real conflict).
        if not mat or difflib.SequenceMatcher(None, mat, C.PIPING_CLASS_TABLE[spec]).ratio() >= 0.6:
            rec["material"] = C.PIPING_CLASS_TABLE[spec]
            prov["material"] = "rule:piping_class_table"
    # BOM: unit vocabulary, size suffix, item numbering
    size = rec.get("nominal_size_in")
    if str(prov.get("nominal_size_in", "")).startswith("vlm:"):
        size = None  # never propagate a VLM-sourced value into other fields
    for i, r in enumerate(rec["bom"], start=1):
        toks = r["description"].split()
        size_tok = toks[-1] if toks and re.fullmatch(r"[0-9ILS|]*[I1L]?N", toks[-1]) else None
        stem = " ".join(toks[:-1] if size_tok else toks)
        m = difflib.get_close_matches(stem, C.BOM_CATALOG, n=1, cutoff=0.6)
        if m:
            stem = m[0]
            r["unit"] = C.BOM_CATALOG[stem]  # catalog is authoritative for the unit
        elif r["unit"] not in C.BOM_UNITS:
            mu = difflib.get_close_matches(r["unit"], C.BOM_UNITS, n=1, cutoff=0.3)
            if mu:
                r["unit"] = mu[0]
        if size is not None:
            r["description"] = f"{stem} {size}IN"
        elif size_tok:
            r["description"] = f"{stem} {size_tok}"
        if r["item"] != i:
            r["item"] = i


def validate(rec: dict) -> set[str]:
    """Return the set of field names that fail schema or cross-field rules."""
    bad = set()
    v = jsonschema.Draft202012Validator(SCHEMA)
    for err in v.iter_errors({k: rec[k] for k in rec if k != "_meta"}):
        bad.add(str(err.path[0]) if err.path else (err.message.split("'")[1] if "'" in err.message else "root"))
    for f in SCALAR_FIELDS:
        if rec.get(f) is None:
            bad.add(f)
    ln = rec.get("line_number")
    if ln and C.LINE_NO_RE.fullmatch(str(ln)):
        size, svc, _, spec = ln.split("-")
        if rec.get("nominal_size_in") is not None and rec["nominal_size_in"] != int(size):
            bad.add("nominal_size_in")
        if rec.get("pipe_spec") and rec["pipe_spec"] != spec:
            bad.add("pipe_spec")
    if rec.get("pipe_spec") in C.PIPING_CLASS_TABLE and rec.get("material") != C.PIPING_CLASS_TABLE[rec["pipe_spec"]]:
        bad.add("material")
    sizes = Counter(m.group(1) for r in rec.get("bom", []) if (m := re.search(r"(\d+)IN$", r.get("description", ""))))
    if sizes and rec.get("nominal_size_in") is not None:
        top, n = sizes.most_common(1)[0]
        if n >= 2 and str(rec["nominal_size_in"]) != top:
            bad.add("nominal_size_in")
    return bad


def run_file(path: Path, cfg: PipelineConfig | None = None) -> dict:
    return extract(Image.open(path), cfg)
