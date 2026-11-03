"""OCR + layout stage (Tesseract via pytesseract, OpenCV for layout)."""
from __future__ import annotations

from dataclasses import dataclass

import os

import cv2
import numpy as np
import pytesseract
from PIL import Image


os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # tesseract's OpenMP threads fight with our process pool
OCR_TIMEOUT_S = float(os.getenv("OCR_TIMEOUT_S", "20"))
TIMEOUTS = {"count": 0}


def _tess(fn, *args, **kw):
    """Run a pytesseract call with a timeout; a timeout counts as an empty read (and is tracked)."""
    try:
        return fn(*args, timeout=OCR_TIMEOUT_S, **kw)
    except RuntimeError as e:  # pytesseract raises RuntimeError on timeout
        if "timeout" in str(e).lower():
            TIMEOUTS["count"] += 1
            return None
        raise


@dataclass
class Word:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int
    line_key: tuple
    angle: int = 0


def to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"))


def binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def prep(gray: np.ndarray, scale: float = 2.0, remove_rules: bool = False) -> np.ndarray:
    """Denoise, upscale and binarize a crop; optionally erase table ruling lines."""
    g = cv2.medianBlur(gray, 3)
    if scale != 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    th = binarize(g)
    if remove_rules:
        inv = 255 - th
        h, w = inv.shape
        lines = cv2.add(
            cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 8), 1))),
            cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, int(0.8 * h))))),
        )
        lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
        th = cv2.bitwise_or(th, lines)
    return th


def ocr_words(gray: np.ndarray, psm: int, angle: int = 0, min_conf: float = 0.0) -> list[Word]:
    data = _tess(pytesseract.image_to_data, gray, config=f"--oem 1 --psm {psm}", output_type=pytesseract.Output.DICT)
    out = []
    if data is None:
        return out
    for i, t in enumerate(data["text"]):
        t = (t or "").strip()
        c = float(data["conf"][i])
        if not t or c < min_conf:
            continue
        out.append(Word(t, c, data["left"][i], data["top"][i], data["width"][i], data["height"][i],
                        (data["block_num"][i], data["par_num"][i], data["line_num"][i]), angle))
    return out


def ocr_lines(gray: np.ndarray, psm: int = 6) -> list[str]:
    words = ocr_words(gray, psm)
    lines: dict[tuple, list[Word]] = {}
    for w in words:
        lines.setdefault(w.line_key, []).append(w)
    return [" ".join(x.text for x in sorted(ws, key=lambda w: w.x)) for _, ws in sorted(lines.items(), key=lambda kv: min(w.y for w in kv[1]))]


def rotate(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=255)


def detect_boxes(gray: np.ndarray, min_area_frac: float = 0.02) -> list[tuple[int, int, int, int]]:
    """Find large ruled rectangles (drawing frame, BOM table, title block)."""
    th = 255 - binarize(gray)
    h, w = gray.shape
    horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (w // 25, 1)))
    vert = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 25)))
    grid = cv2.dilate(cv2.add(horiz, vert), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh > min_area_frac * w * h and bw < 0.97 * w:
            boxes.append((x, y, x + bw, y + bh))
    # de-duplicate near-identical boxes (inner/outer contour of the same thick line)
    uniq: list[tuple[int, int, int, int]] = []
    for b in sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True):
        if not any(all(abs(b[i] - u[i]) < 20 for i in range(4)) for u in uniq):
            uniq.append(b)
    return uniq


def crop(gray: np.ndarray, box, pad: int = 4) -> np.ndarray:
    x0, y0, x1, y1 = box
    return gray[max(0, y0 + pad): y1 - pad, max(0, x0 + pad): x1 - pad]


def table_rows(gray: np.ndarray, min_row_h: int = 18, preprocess: bool = True) -> list[str]:
    """Split a ruled table into row bands using its horizontal rules, OCR each band as one line."""
    inv = 255 - binarize(gray)
    h, w = inv.shape
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 3), 1)))
    ys = np.where(horiz.sum(axis=1) > 0.5 * 255 * w)[0]
    cuts = [0]
    for y in ys:
        if y - cuts[-1] > 3:
            cuts.append(int(y))
        else:
            cuts[-1] = int(y)
    cuts.append(h)
    lines = []
    for y0, y1 in zip(cuts[:-1], cuts[1:]):
        if y1 - y0 < min_row_h:
            continue
        band = gray[y0 + 3: y1 - 2, :]
        if band.size == 0 or (255 - band).max() < 80:  # empty row
            continue
        g = prep(band, remove_rules=True) if preprocess else band
        g = cv2.copyMakeBorder(g, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        txt = (_tess(pytesseract.image_to_string, g, config="--oem 1 --psm 7") or "").strip()
        if txt:
            lines.append(txt)
    return lines


def deskew(gray: np.ndarray, max_angle: float = 5.0) -> tuple[np.ndarray, float]:
    """Estimate page skew from long near-horizontal lines (sheet frame, table rules) and undo it."""
    inv = 255 - binarize(gray)
    h, w = inv.shape
    lines = cv2.HoughLinesP(inv, 1, np.pi / 1800, threshold=300, minLineLength=w // 4, maxLineGap=10)
    if lines is None:
        return gray, 0.0
    angs = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < max_angle:
            angs.append(a)
    if not angs:
        return gray, 0.0
    a = float(np.median(angs))
    if abs(a) < 0.05:
        return gray, 0.0
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255), a
