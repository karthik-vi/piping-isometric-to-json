"""Synthetic piping-isometric generator.

Draws a simplified isometric pipe route with rotated dimension text,
instrument/valve tags, a BOM table and a title block, and writes the exact
ground-truth JSON next to it. These are SYNTHETIC drawings, not client data.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 2000, 1400
DRAW_AREA = (40, 40, 1300, 1360)  # x0, y0, x1, y1 for the pipe route
BOM_BOX = (1340, 40, 1960, 700)
TITLE_BOX = (1340, 960, 1960, 1360)

SERVICES = ["P", "CW", "ST", "N", "IA", "HC"]
SPECS = ["A1A", "A3B", "B1A", "C2A", "D1B", "S4A"]
MATERIALS = {
    "A1A": "CS A106-B",
    "A3B": "CS A333-6",
    "B1A": "SS A312-TP316L",
    "C2A": "CS GALV A53-B",
    "D1B": "DUPLEX A790-S31803",
    "S4A": "SS A312-TP304",
}
SIZES = [1, 2, 3, 4, 6, 8, 10, 12]
TAG_PREFIXES = ["HV", "PI", "TI", "FT", "PSV", "XV", "LT", "TW"]

# Six isometric directions (unit vectors in screen space): +/-x at 30deg, +/-y at 150deg, +/-z vertical.
_C, _S = math.cos(math.radians(30)), math.sin(math.radians(30))
ISO_DIRS = {
    "E": (_C, -_S),
    "W": (-_C, _S),
    "N": (-_C, -_S),
    "S": (_C, _S),
    "UP": (0.0, -1.0),
    "DN": (0.0, 1.0),
}

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def _font(size: int) -> ImageFont.ImageFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size=size)


@dataclass
class GenConfig:
    seed: int = 0
    scanned: bool = False  # add scan-like degradation (noise, blur, skew)
    n_segments: tuple[int, int] = (4, 8)
    n_tags: tuple[int, int] = (2, 5)


def _rand_record(rng: random.Random) -> dict:
    size = rng.choice(SIZES)
    service = rng.choice(SERVICES)
    spec = rng.choice(SPECS)
    seq = rng.randint(1000, 9999)
    line_number = f"{size}-{service}-{seq}-{spec}"
    return {
        "drawing_number": f"ISO-{rng.randint(100, 999)}-{rng.randint(1, 99):02d}",
        "line_number": line_number,
        "nominal_size_in": size,
        "service": service,
        "pipe_spec": spec,
        "material": MATERIALS[spec],
        "design_pressure_barg": round(rng.choice([3.5, 7.0, 10.0, 15.0, 19.6, 25.0, 40.0, 51.1]), 1),
        "design_temp_c": rng.choice([40, 65, 80, 120, 150, 180, 250, 350]),
        "revision": rng.choice(["0", "A", "B", "1", "2"]),
    }


def _bom(rng: random.Random, size: int) -> list[dict]:
    items = [
        ("PIPE SMLS BE", "M", (6, 40)),
        ("ELBOW 90 LR", "EA", (2, 8)),
        ("FLANGE WN RF", "EA", (2, 6)),
        ("GASKET SPW", "EA", (2, 6)),
        ("GATE VALVE", "EA", (1, 2)),
        ("TEE EQUAL", "EA", (1, 3)),
        ("STUD BOLT", "SET", (2, 6)),
        ("REDUCER CONC", "EA", (1, 2)),
    ]
    rng.shuffle(items)
    rows = []
    for i, (desc, unit, (lo, hi)) in enumerate(items[: rng.randint(4, 7)], start=1):
        rows.append({"item": i, "qty": rng.randint(lo, hi), "unit": unit, "description": f"{desc} {size}IN"})
    return rows


def _rotated_text(img: Image.Image, xy: tuple[float, float], text: str, angle: float, font):
    """Paste `text` rotated by `angle` degrees (counter-clockwise) centred at xy."""
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8
    layer = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(layer).text((4 - bbox[0], 4 - bbox[1]), text, fill=255, font=font)
    layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
    x, y = int(xy[0] - layer.width / 2), int(xy[1] - layer.height / 2)
    img.paste(Image.new("L", layer.size, 0), (x, y), layer)
    return (x, y, x + layer.width, y + layer.height)


def _rotated_bbox(xy, text, angle, font):
    bbox = font.getbbox(text)
    tw, th = bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 8
    a = math.radians(angle)
    w = abs(tw * math.cos(a)) + abs(th * math.sin(a))
    h = abs(tw * math.sin(a)) + abs(th * math.cos(a))
    return (xy[0] - w / 2, xy[1] - h / 2, xy[0] + w / 2, xy[1] + h / 2)


def _draw_valve(d: ImageDraw.ImageDraw, c, u, s=14):
    ux, uy = u
    px, py = -uy, ux
    a = (c[0] - ux * s + px * s * 0.7, c[1] - uy * s + py * s * 0.7)
    b = (c[0] - ux * s - px * s * 0.7, c[1] - uy * s - py * s * 0.7)
    e = (c[0] + ux * s + px * s * 0.7, c[1] + uy * s + py * s * 0.7)
    f = (c[0] + ux * s - px * s * 0.7, c[1] + uy * s - py * s * 0.7)
    d.polygon([a, b, c, f, e, c], outline=0, fill=255, width=2)


def _route_clear(pts, min_gap=110.0) -> bool:
    """Reject routes whose non-adjacent segment midpoints/vertices come too close."""
    probes = []
    for a, b in zip(pts[:-1], pts[1:]):
        for t in (0.0, 0.25, 0.5, 0.75):
            probes.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    probes.append(pts[-1])
    for i in range(len(probes)):
        for j in range(i + 3, len(probes)):
            if math.dist(probes[i], probes[j]) < min_gap:
                return False
    return True


def _overlaps(box, boxes, pad=6) -> bool:
    x0, y0, x1, y1 = box
    return any(not (x1 + pad < b[0] or b[2] + pad < x0 or y1 + pad < b[1] or b[3] + pad < y0) for b in boxes)


def _route(rng: random.Random, n: int):
    """Random walk on the isometric grid that stays inside the drawing area."""
    x0, y0, x1, y1 = DRAW_AREA
    for _ in range(400):
        pts = [(rng.uniform(x0 + 250, x1 - 250), rng.uniform(y0 + 250, y1 - 250))]
        dirs, lens = [], []
        prev = None
        ok = True
        for _ in range(n):
            choices = [k for k in ISO_DIRS if k != prev and (prev is None or ISO_DIRS[k] != tuple(-v for v in ISO_DIRS[prev]))]
            k = rng.choice(choices)
            length_mm = rng.choice(range(600, 4800, 50))
            px_len = 150 + length_mm / 4800 * 190
            dx, dy = ISO_DIRS[k]
            nx, ny = pts[-1][0] + dx * px_len, pts[-1][1] + dy * px_len
            if not (x0 + 30 < nx < x1 - 30 and y0 + 30 < ny < y1 - 30):
                ok = False
                break
            pts.append((nx, ny))
            dirs.append(k)
            lens.append(length_mm)
            prev = k
        if ok and _route_clear(pts):
            return pts, dirs, lens
    raise RuntimeError("could not place route")


def _table(d, box, header, rows, font, col_x):
    x0, y0, x1, y1 = box
    d.rectangle(box, outline=0, width=3)
    row_h = 44
    y = y0 + 10
    for i, h in enumerate(header):
        d.text((x0 + col_x[i], y), h, fill=0, font=font)
    y += row_h
    d.line((x0, y - 6, x1, y - 6), fill=0, width=2)
    for r in rows:
        for i, v in enumerate(r):
            d.text((x0 + col_x[i], y), str(v), fill=0, font=font)
        y += row_h
        d.line((x0, y - 6, x1, y - 6), fill=0, width=1)


def generate(cfg: GenConfig) -> tuple[Image.Image, dict]:
    rng = random.Random(cfg.seed)
    rec = _rand_record(rng)
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    d.rectangle((10, 10, W - 10, H - 10), outline=0, width=4)
    d.rectangle(DRAW_AREA, outline=0, width=1)

    n_seg = rng.randint(*cfg.n_segments)
    while True:
        try:
            pts, dirs, lens = _route(rng, n_seg)
            break
        except RuntimeError:
            if n_seg <= 3:
                raise
            n_seg -= 1
    d.line(pts, fill=0, width=5, joint="curve")
    for p in pts[1:-1]:  # elbows
        d.ellipse((p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7), outline=0, width=2)

    dim_font = _font(26)
    dimensions = []
    placed: list[tuple] = []
    for a, b in zip(pts[:-1], pts[1:]):  # pipe geometry is an obstacle for labels
        n = max(2, int(math.dist(a, b) / 12))
        for i in range(n + 1):
            x, y = a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n
            placed.append((x - 6, y - 6, x + 6, y + 6))
    for (a, b), k, L in zip(zip(pts[:-1], pts[1:]), dirs, lens):
        dx, dy = ISO_DIRS[k]
        mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        ang = -math.degrees(math.atan2(dy, dx))
        if ang > 90:
            ang -= 180
        if ang < -90:
            ang += 180
        for off in (34, -34, 60, -60, 90, -90):
            pos = (mid[0] + dy * off, mid[1] - dx * off)  # perpendicular offset
            bb = _rotated_bbox(pos, str(L), ang, dim_font)
            if not _overlaps(bb, placed):
                break
        placed.append(_rotated_text(img, pos, str(L), ang, dim_font))
        dimensions.append(L)

    # valves / instruments with tags
    tag_font = _font(28)
    tags = []
    seg_ids = rng.sample(range(len(dirs)), k=min(len(dirs), rng.randint(*cfg.n_tags)))
    used = set()
    for si in seg_ids:
        a, b = pts[si], pts[si + 1]
        t = rng.uniform(0.25, 0.75)
        c = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
        _draw_valve(d, c, ISO_DIRS[dirs[si]])
        while True:
            tag = f"{rng.choice(TAG_PREFIXES)}-{rng.randint(100, 9999)}"
            if tag not in used:
                used.add(tag)
                break
        tb_ = tag_font.getbbox(tag)
        tw = tb_[2] - tb_[0]
        for _ in range(60):
            tx = c[0] + rng.choice([-1, 1]) * rng.uniform(80, 170)
            ty = c[1] + rng.choice([-1, 1]) * rng.uniform(60, 140)
            ly = ty - 34 if ty < c[1] else ty + 2
            box = (tx - 50, ly, tx - 50 + tw, ly + 34)
            x0_, y0_, x1_, y1_ = DRAW_AREA
            if box[0] > x0_ + 5 and box[2] < x1_ - 5 and box[1] > y0_ + 5 and box[3] < y1_ - 5 and not _overlaps(box, placed, pad=12):
                break
        placed.append(box)
        # leader stops a few px short of the label, as in normal drafting practice
        end = ((box[0] + box[2]) / 2, box[3] + 4 if ty < c[1] else box[1] - 4)
        d.line((c, end), fill=0, width=1)
        d.text((tx - 50, ly), tag, fill=0, font=tag_font)
        tags.append(tag)

    # flow arrow + line label along first segment (rotated, like real isos)
    k0 = dirs[0]
    dx, dy = ISO_DIRS[k0]
    ang = -math.degrees(math.atan2(dy, dx))
    ang = ang - 180 if ang > 90 else (ang + 180 if ang < -90 else ang)
    mid0 = ((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
    lf = _font(24)
    for off in (40, -40, 70, -70, 100, -100):
        pos = (mid0[0] - dy * off, mid0[1] + dx * off)
        if not _overlaps(_rotated_bbox(pos, rec["line_number"], ang, lf), placed):
            break
    placed.append(_rotated_text(img, pos, rec["line_number"], ang, lf))

    # BOM
    bom = _bom(rng, rec["nominal_size_in"])
    _table(d, BOM_BOX, ["ITEM", "QTY", "UNIT", "DESCRIPTION"],
           [[r["item"], r["qty"], r["unit"], r["description"]] for r in bom], _font(26), [12, 90, 170, 260])

    # title block
    tb = [
        ("DWG NO", rec["drawing_number"]),
        ("LINE NO", rec["line_number"]),
        ("SIZE", f"{rec['nominal_size_in']} IN"),
        ("SPEC", rec["pipe_spec"]),
        ("MATERIAL", rec["material"]),
        ("DESIGN PRESS", f"{rec['design_pressure_barg']} BARG"),
        ("DESIGN TEMP", f"{rec['design_temp_c']} C"),
        ("REV", rec["revision"]),
    ]
    x0, y0, x1, y1 = TITLE_BOX
    d.rectangle(TITLE_BOX, outline=0, width=3)
    f = _font(26)
    for i, (k, v) in enumerate(tb):
        y = y0 + 10 + i * 48
        d.text((x0 + 12, y), f"{k}:", fill=0, font=f)
        d.text((x0 + 250, y), v, fill=0, font=f)
        d.line((x0, y + 40, x1, y + 40), fill=0, width=1)
    d.text((40, H - 40), "SYNTHETIC DRAWING - NOT FOR CONSTRUCTION", fill=0, font=_font(18))

    if cfg.scanned:
        img = _degrade(img, rng)

    gt = dict(rec)
    gt["bom"] = bom
    gt["tags"] = sorted(tags)
    gt["dimensions_mm"] = sorted(dimensions)
    return img, gt


def _degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.rotate(rng.uniform(-1.2, 1.2), resample=Image.BICUBIC, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.1)))
    arr = np.asarray(img).astype(np.float32)
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    arr += nrng.normal(0, rng.uniform(8, 18), arr.shape)
    speck = nrng.random(arr.shape) < 0.002
    arr[speck] = 0
    arr = arr * rng.uniform(0.85, 1.0) + rng.uniform(0, 25)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def write_dataset(out_dir: Path, n: int, seed: int, scanned_frac: float = 0.5) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    paths = []
    for i in range(n):
        scanned = rng.random() < scanned_frac
        img, gt = generate(GenConfig(seed=seed * 100_000 + i, scanned=scanned))
        stem = f"iso_{i:04d}{'_scan' if scanned else ''}"
        img.save(out_dir / f"{stem}.png")
        (out_dir / f"{stem}.json").write_text(json.dumps(gt, indent=2))
        paths.append(out_dir / f"{stem}.png")
    return paths
