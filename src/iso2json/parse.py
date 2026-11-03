"""Rule-based parsing of OCR text into typed fields."""
from __future__ import annotations

import difflib
import re

from . import config as C

TITLE_LABELS = {
    "DWG NO": "drawing_number",
    "LINE NO": "line_number",
    "SIZE": "nominal_size_in",
    "SPEC": "pipe_spec",
    "MATERIAL": "material",
    "DESIGN PRESS": "design_pressure_barg",
    "DESIGN TEMP": "design_temp_c",
    "REV": "revision",
}


def _split_label(line: str) -> tuple[str | None, str]:
    """Return (canonical field, raw value) for a title-block line."""
    line = line.strip()
    if ":" in line:
        lab, val = line.split(":", 1)
    else:  # colon lost by OCR: try every prefix length that matches a label
        best = (None, 0.0, line)
        toks = line.split()
        for k in range(1, min(3, len(toks)) + 1):
            cand = " ".join(toks[:k])
            m = difflib.get_close_matches(cand.upper(), TITLE_LABELS, n=1, cutoff=0.7)
            if m:
                r = difflib.SequenceMatcher(None, cand.upper(), m[0]).ratio()
                if r > best[1]:
                    best = (m[0], r, " ".join(toks[k:]))
        if best[0] is None:
            return None, line
        return TITLE_LABELS[best[0]], best[2].strip()
    m = difflib.get_close_matches(lab.strip().upper(), TITLE_LABELS, n=1, cutoff=0.6)
    return (TITLE_LABELS[m[0]] if m else None), val.strip()


def fix_line_number(s: str) -> str | None:
    s = s.upper().replace(" ", "").replace("—", "-").replace("_", "-")
    parts = s.split("-")
    if len(parts) != 4:
        m = C.LINE_NO_RE.search(s)
        return m.group(0) if m else None
    size, svc, seq, spec = parts
    size = size.translate(C.DIGIT_FIX)
    svc = svc.translate(C.ALPHA_FIX)
    seq = seq.translate(C.DIGIT_FIX)
    if len(spec) == 3:
        spec = spec[0].translate(C.ALPHA_FIX) + spec[1].translate(C.DIGIT_FIX) + spec[2].translate(C.ALPHA_FIX)
    cand = f"{size}-{svc}-{seq}-{spec}"
    return cand if C.LINE_NO_RE.fullmatch(cand) else None


def _num(s: str, as_int: bool = False):
    s = s.upper().translate(C.DIGIT_FIX)
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    v = float(m.group(0))
    return int(round(v)) if as_int else v


def parse_title_block(lines: list[str]) -> dict:
    out: dict = {}
    for line in lines:
        field, val = _split_label(line)
        if not field or field in out:
            continue
        v = val.upper().strip()
        if field == "drawing_number":
            v = v.replace(" ", "")
            v = re.sub(r"^[^-]*?[1IL|]?[S5][O0D]-", "ISO-", v)  # 1S0-, 1ISO-, IS0- ...
            if v.startswith("ISO-"):
                v = "ISO-" + v[4:].translate(C.DIGIT_FIX)
            out[field] = v if C.DWG_NO_RE.fullmatch(v) else v or None
        elif field == "line_number":
            out[field] = fix_line_number(v) or v
        elif field == "nominal_size_in":
            n = _num(v)
            out[field] = (int(n) if n is not None and float(n).is_integer() else n)
        elif field == "pipe_spec":
            v = v.replace(" ", "")[:3]
            if len(v) == 3:
                v = v[0].translate(C.ALPHA_FIX) + v[1].translate(C.DIGIT_FIX) + v[2].translate(C.ALPHA_FIX)
            out[field] = v
        elif field == "material":
            out[field] = re.sub(r"\s+", " ", v)
        elif field == "design_pressure_barg":
            out[field] = _num(v)
        elif field == "design_temp_c":
            out[field] = _num(v, as_int=True)
        elif field == "revision":
            out[field] = v.split()[0] if v.split() else None
    return out


BOM_ROW_RE = re.compile(r"^(\S+)\s+(\S+)\s+(\S+)\s+(.+)$")


def parse_bom(lines: list[str]) -> list[dict]:
    rows = []
    for line in lines:
        if "DESCRIPTION" in line.upper() or "QTY" in line.upper():
            continue
        m = BOM_ROW_RE.match(line.strip())
        if not m:
            continue
        item, qty, unit, desc = m.groups()
        item_n, qty_n = _num(item, True), _num(qty, True)
        unit = unit.upper().translate(C.ALPHA_FIX)
        if item_n is None or qty_n is None:
            continue
        desc = re.sub(r"[^A-Z0-9 \-]", "", desc.upper())
        rows.append({"item": item_n, "qty": qty_n, "unit": unit, "description": re.sub(r"\s+", " ", desc).strip()})
    return rows


def clean_token(t: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]", "", t).upper()


def parse_tag(t: str, fuzzy_prefix: bool = True) -> str | None:
    t = clean_token(t)
    if "-" not in t:
        m = re.fullmatch(r"([A-Z]{2,3})(\d{3,4})", t)  # dash lost by OCR
        if not m:
            return None
        t = f"{m.group(1)}-{m.group(2)}"
    pre, num = t.split("-", 1)
    pre = pre.translate(C.ALPHA_FIX)
    if fuzzy_prefix and pre not in C.TAG_PREFIXES and 2 <= len(pre) <= 3:
        same_len = [p for p in C.TAG_PREFIXES if len(p) == len(pre)]
        m = difflib.get_close_matches(pre, same_len, n=1, cutoff=0.5)
        if m:
            pre = m[0]
    cand = f"{pre}-{num.translate(C.DIGIT_FIX)}"
    return cand if C.TAG_RE.fullmatch(cand) else None


def parse_dim(t: str) -> int | None:
    t = clean_token(t)
    if not C.DIM_RE.fullmatch(t):
        return None
    v = int(t)
    lo, hi = C.DIM_RANGE_MM
    return v if lo <= v <= hi else None
