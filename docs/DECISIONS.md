# Architecture Decision Records

## Period freeze

| | |
|---|---|
| Work period reconstructed | Marvel Technology Solutions, AI/ML Engineer Intern, Jun 2026 – present |
| Cutoff | Nov 2026 |
| Python | 3.12 |
| OCR engine | Tesseract 5.3.4 (Ubuntu 24.04 package), via pytesseract 0.3.13 |
| Image / layout | OpenCV (opencv-python-headless 5.0.0.93, PyPI upload 2026-07-02), Pillow 13.0.0 (2026-10-16), NumPy 2.5.4 (2026-10-10) |
| Validation | jsonschema 4.26.0, JSON Schema draft 2020-12 |
| Local VLM | `HuggingFaceTB/SmolVLM-256M-Instruct` (HF repo created 2025-01-17, Apache-2.0) via transformers 5.22.1 (2026-10-30) + torch 2.15.0 CPU (2026-10-28) |
| API VLM (optional, untested) | Anthropic Messages API, default `claude-sonnet-4-5` (released 2025-09-29), overridable via `ANTHROPIC_MODEL` |

All versions were checked against PyPI upload dates / Hugging Face `createdAt` on 2026-11-03; every one predates the
Nov 2026 cutoff. No exceptions needed.

---

## ADR-001 Data: synthetic generator instead of real drawings — 2026-11-03

- **Context.** The original work used client drawings, which cannot be used or shown. No public labelled isometric dataset found (see DATA.md).
- **Options.** (a) Scrape CAD-block sites for sample isometrics and hand-label a few; (b) synthetic generator with exact labels; (c) P&ID datasets.
- **Decision.** (b). Exact field-level ground truth, any number of samples, controllable degradation, no licensing risk.
- **Tradeoffs.** Distribution is narrow (one template/font/vocabulary). Numbers are an upper bound on what the same pipeline would do on real drawings.
- **Would change my mind.** A licensed set of ≥50 real isometrics, even unlabelled. I would hand-label 30 for a real test split.

## ADR-002 OCR engine: Tesseract 5 (LSTM) — 2026-11-03

- **Context.** Need a free, offline, CPU OCR that runs in CI.
- **Options.** Tesseract 5; TrOCR (needs line crops, and is slower on CPU); PaddleOCR / docTR (heavier installs); cloud OCR (keys, cost).
- **Decision.** Tesseract with region-specific page-segmentation modes: `psm 7` per table row, `psm 11` (sparse) for the drawing area.
- **Tradeoffs.** Tesseract cannot read rotated text. Handled by rotating the drawing area by ±30° and ±90° (isometric text angles), which costs 5× OCR time on that region.
  Unpreprocessed noisy pages make `psm 11` pathologically slow (minutes per page). Fixed with a per-call timeout and denoise/binarize.
- **Would change my mind.** If text angles were arbitrary (not isometric), a text detector (e.g. DB/EAST) plus a recognizer on rectified crops would win.

## ADR-003 Rules before VLM, and the VLM is gated by validation — 2026-11-03

- **Context.** VLMs can read hard crops but can also produce plausible wrong values. In engineering data a confident wrong value is worse than a blank.
- **Options.** (a) VLM-first end-to-end JSON; (b) OCR+rules, VLM as fallback for failing fields, answer accepted only if it passes the same validation; (c) no VLM.
- **Decision.** (b). Every field has provenance (`ocr`, `rule:*`, `vlm:*`) in `_meta.provenance`; fields still failing land in `_meta.needs_review`.
- **Tradeoffs.** The VLM is only consulted for fields that the validator can recognise as wrong. Tags/dimensions that are silently wrong are not sent to it.
- **Would change my mind.** A VLM that is cheaper than the OCR stack and measurably more accurate on the held-out split. Not true for SmolVLM-256M on CPU.

## ADR-004 Domain tables in config, not code — 2026-11-03

- Piping class → material, service codes, tag prefixes, BOM catalog (stem → unit) live in `src/iso2json/config.py`.
- Real projects have these as the piping material specification and item catalog. The repair stage uses them for snapping.
- Risk acknowledged: on synthetic data the tables exactly match the generator, which inflates repair gains.

## ADR-005 Evaluation: field exact match + set F1, dev/test split by seed — 2026-11-03

- **Metrics.** Per-field exact match (scalar fields); set precision/recall/F1 for tags, dimensions, BOM rows (a BOM row counts only if item, qty, unit and description all match);
  drawing-level exact match (every field right); error-flag rate (share of wrong drawings that the validator flagged).
- **Protocol.** All tuning was done on `dev` (seed 1, 30 drawings). `test` (seed 2, 60 drawings) was generated before tuning and is only used for the reported tables.
- **Why not character error rate.** Downstream systems consume fields, not characters. One wrong digit in a line number is a wrong record.

## ADR-006 Deployment: batch CLI + Docker, no API — 2026-11-03

- Isometric extraction is a batch job (drawing packages arrive in bulk). A CLI with an exit code that signals "needs review" is enough for the demo.
- Docker image excludes torch to stay small; VLM extras are opt-in.
- Would add a FastAPI wrapper + queue only if per-drawing interactive use were needed.
