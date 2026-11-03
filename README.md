# piping-isometric-to-json

Multi-stage pipeline that turns a piping isometric drawing (PNG) into validated, structured JSON:
OCR → rule-based parsing/filtering → cross-field validation & repair → optional Vision-Language-Model re-read of fields that still fail.

## What this is / is not

| | |
|---|---|
| **Reconstructs** | Resume bullet (Marvel Technology Solutions, AI/ML Engineer Intern, Jun 2026–present): *"Engineered a multi-stage pipeline fusing OCR, rule-based filtering, and Vision-Language Models to convert piping isometrics into structured JSON, achieving 95%+ exact match and 97% F1."* (verbatim from the resume, the source of truth) |
| **Is** | A portfolio reconstruction written from scratch in this repo, run on **synthetic** isometrics produced by `src/iso2json/generator.py`. All metrics below were measured here (`docs/EXPERIMENTS.md`). |
| **Is not** | Marvel's code, Marvel's or any client's drawings, or a production system. No client name, drawing, spec table or internal tool is used. The piping-class table, tag prefixes and BOM catalog are generic, invented values. |
| **Data** | Synthetic only (no public labelled isometric dataset was found; see `docs/DATA.md`). |

### Period freeze

Work period Jun 2026 – present, **cutoff Nov 2026**. Python 3.12; Tesseract 5.3.4; OpenCV 5.0.0.93; Pillow 13.0.0; NumPy 2.5.4;
jsonschema 4.26.0; optional VLM: `HuggingFaceTB/SmolVLM-256M-Instruct` (published 2025-01-17) on transformers 5.22.1 / torch 2.15.0 (CPU);
optional API VLM: Anthropic `claude-sonnet-4-5` (released 2025-09-29). Every version predates the cutoff (release dates checked on PyPI/HF,
see `docs/DECISIONS.md`). No exceptions.

## Problem

Isometric drawings carry the data that construction and material take-off need (line number, spec, material,
design conditions, BOM, valve/instrument tags, cut dimensions), but they arrive as images or flattened PDFs.
Pure OCR misses rotated text and confuses characters. Pure VLM extraction is slow and can return confident wrong values.
The goal: high field-level accuracy, every value traceable to a stage, and anything uncertain routed to a human.

## Architecture

```
drawing.png
   │
   ├─ 0 preprocess   deskew (Hough on frame/table rules) ─ median denoise ─ 2× upscale ─ Otsu
   ├─ 1 layout       ruled-box detection → drawing area | BOM table | title block   (template fallback)
   ├─ 2 OCR          title/BOM: split into ruled rows, Tesseract psm 7 per row (rules erased)
   │                 drawing:  Tesseract psm 11 at 0°, ±30°, ±90° (isometric text angles) + raw 0° read
   ├─ 3 rules        field grammars + OCR-confusion maps (O↔0, I↔1 …) per field type
   │                 tags: fuzzy prefix, fragment merge, IoU de-dup; dims: vote / confidence / substring filter
   ├─ 4 repair       line no ↔ size/service/spec; spec → material (piping-class table); BOM catalog; item numbering
   ├─   validate     JSON Schema (schemas/isometric.schema.json) + cross-field rules → needs_review set
   └─ 5 VLM (opt.)   only for fields in needs_review; answer parsed by the same grammar and accepted
                     only if validation then passes (none | smolvlm | anthropic)
   ▼
validated.json   (+ _meta.provenance per field, _meta.needs_review, latency)
```

## Quickstart (< 10 min)

```bash
sudo apt-get install -y tesseract-ocr fonts-dejavu-core      # macOS: brew install tesseract
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_pipeline.py data/samples/iso_clean_0001.png -o out/iso_clean_0001.json
pytest -q

# reproduce the tables (≈15 min on 4 CPUs)
python scripts/generate_dataset.py --out data/synthetic/test --n 60 --seed 2
python scripts/evaluate.py --data data/synthetic/test --variants A,B,C,D,E,E2 --out results/test_ablation.json
```
Optional VLM stage: `pip install -r requirements-vlm.txt` then `--vlm smolvlm` (local, no key) or `--vlm anthropic` (needs `ANTHROPIC_API_KEY`).
Docker: `docker build -t iso2json . && docker run --rm iso2json`.

## Results (held-out synthetic test set, seed 2, n=60: 36 clean + 24 scanned)

| Variant | Field EM | Drawing EM | Tags F1 | Dims F1 | BOM F1 |
|---|---|---|---|---|---|
| A full-page OCR + regex (baseline) | 0.002 | 0.000 | 0.928 | 0.009 | 0.000 |
| C + layout, preprocessing, row-wise tables | 0.898 | 0.000 | 0.910 | 0.021 | 0.819 |
| D + multi-rotation OCR | 0.898 | 0.200 | 0.910 | 0.952 | 0.819 |
| **E2 full rules (default, no VLM)** | **0.930** | **0.550** | **0.950** | **0.953** | **0.920** |
| F E2 + guarded SmolVLM-256M re-read (opt-in) | 0.935 | 0.567 | 0.950 | 0.953 | 0.920 |

Clean drawings: field EM 1.000, drawing EM 0.722. Scanned: field EM 0.824, drawing EM 0.292. Full table, VLM run and
per-field numbers: `docs/EXPERIMENTS.md`.

> Resume claim: 95%+ exact match | Reproduced here on synthetic isometrics (seed 2, n=60): 93.0% field-level exact match (100.0% clean / 82.4% scanned), 55.0% drawing-level exact match | Gap / why: original data and metric definition unavailable; synthetic scans are noisier than vector PDFs; drawing-level EM is far stricter than field-level.

> Resume claim: 97% F1 | Reproduced here on the same synthetic test set (n=60), variant E2: tags F1 95.0%, dimensions F1 95.3%, BOM-row F1 92.0% (no single pooled F1 is computed; the resume's F1 definition is not available) | Gap / why: same as above; the original F1 definition (which fields, micro vs macro) is unknown, so only per-field F1 is reported.

### Failure cases (E2, test)

- **Dimensions (17 drawings wrong).** A dimension next to a valve symbol or a leader line is not read in any rotation; a missing value
  passes validation because nothing tells the pipeline how many dimensions to expect.
- **Tags (11).** Prefix destroyed on scans (`|-3964` for `PI-3964`); occasionally a digit swap (`FT-3394` for `FT-5394`) that passes the tag grammar.
- **Title block on scans.** `drawing_number`, `revision`, design temp/pressure are short tokens with no cross-field check, so OCR slips there
  (`ISO-174-691`, `REV 20`) are only caught when the grammar breaks.
- **VLM hallucination (EXP-005).** Unguarded, SmolVLM answered "16" for several fields; the size was accepted and propagated into
  every BOM row. Guards added in EXP-006 (no VLM value feeds repairs, BOM-size cross-check, duplicate-answer rejection).
- **Validator coverage.** It flags 37% of wrong drawings and 0% of correct ones. `needs_review` is trustworthy; `valid=true` is not a guarantee.

## Limitations

- Synthetic data, one template, one font, a closed vocabulary that the rules also know (DATA.md). Expect lower accuracy on real drawings.
- The Anthropic VLM path is implemented but **untested** here (no API key in this environment).
- No symbol detection: valves/elbows are not extracted as components, only their tags.
- CPU latency ≈ 9 s per drawing (median), dominated by 6 Tesseract passes over the drawing area.

## How to extend

- Real drawings: add a loader for vector PDFs (text layer first, OCR as fallback) and hand-label 30 drawings as a real test split.
- Add pipe-segment detection to predict the expected count of dimensions and valve symbols, turning silent misses into `needs_review`.
- Swap Tesseract for a text detector + recognizer on rectified crops if text angles are not limited to isometric axes.
- Domain tables (`config.py`) should be loaded from the project's piping material spec instead of hard-coded.
