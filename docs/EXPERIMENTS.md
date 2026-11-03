# Experiment log (newest first)

All numbers below were produced by `scripts/evaluate.py` in this repo on 2026-11-03, on a 4-vCPU / no-GPU container.
Raw outputs: `results/test_ablation.json`, `results/test_vlm.json`, `results/dev.json`, `results/dev_e2.json`.

Datasets (see DATA.md): **dev** = synthetic, seed 1, 30 drawings. **test** = synthetic, seed 2, 60 drawings (36 clean, 24 "scanned").
Test was generated before any tuning and used only for the final runs below.

Metric definitions: *field EM* = micro exact match over the 9 scalar fields; *drawing EM* = all 12 fields
(9 scalars + tags + dimensions + BOM) exactly right; *set F1* = micro F1 over tags / dimensions / BOM rows; *err-flag* = share of
wrong drawings that the validator put in `needs_review`.

---

## EXP-005 — VLM re-read of flagged fields (SmolVLM-256M, local CPU) — test

- **Hypothesis.** Fields that still fail validation after the rules are often readable by a small VLM; gated acceptance means it cannot make things worse.
- **Config.** Variant F = E2 + `vlm_provider=smolvlm`, model `HuggingFaceTB/SmolVLM-256M-Instruct`, greedy decoding, max 24 new tokens, one question per failing title-block field. Single worker.
- **Result.** See `results/test_vlm.json`; summary table is added to the README *Results* section once this run finished (this entry is updated in the same commit).

## EXP-004 — Full ablation on the held-out test set — test (seed 2, n=60)

- **Hypothesis.** Each stage adds measurable accuracy; layout + preprocessing matter most for title block/BOM, multi-rotation OCR for dimensions.
- **Config.** Variants A → E2 as defined in `scripts/evaluate.py` (cumulative). Tesseract 5.3.4, `--oem 1`, OCR timeout 20 s.

| Variant | Stage added | Field EM | Drawing EM | Tags F1 | Dims F1 | BOM F1 | Err-flag | Median s/drawing |
|---|---|---|---|---|---|---|---|---|
| A | full-page OCR + regex (baseline) | 0.002 | 0.000 | 0.928 | 0.009 | 0.000 | 1.00 | 1.4 |
| B | + layout crops (ruled-box detection) | 0.733 | 0.000 | 0.895 | 0.010 | 0.629 | 0.47 | 0.8 |
| C | + deskew / denoise / 2× upscale + row-wise table OCR | 0.898 | 0.000 | 0.910 | 0.021 | 0.819 | 0.42 | 5.1 |
| D | + multi-rotation (0, ±30°, ±90°) drawing OCR + dim filter | 0.898 | 0.200 | 0.910 | 0.952 | 0.819 | 0.52 | 7.8 |
| E | + cross-field repair (line no ↔ size/spec, class table, BOM catalog) | 0.930 | 0.433 | 0.910 | 0.952 | 0.920 | 0.29 | 8.2 |
| **E2** | + tag rules (fuzzy prefix, fragment merge, dual read) | **0.930** | **0.533** | **0.948** | **0.953** | **0.920** | 0.36 | 8.9 |

Clean vs scanned for E2: clean (n=36) field EM 1.000, drawing EM 0.694, tags F1 0.988, dims F1 0.978, BOM F1 0.990;
scanned (n=24) field EM 0.824, drawing EM 0.292, tags F1 0.880, dims F1 0.913, BOM F1 0.813.

E2 per-field exact match: drawing_number 0.850, line_number 0.950, nominal_size_in 0.950, service 0.950, pipe_spec 0.983,
material 0.983, design_pressure 0.917, design_temp 0.900, revision 0.883, tags 0.817, dimensions 0.717, BOM 0.833.

- **What surprised me.**
  - The baseline's tag F1 (0.928) is *higher* than B's (0.895): on the full page, the sparse-text pass sees tags at native resolution,
    while B (no preprocessing) reads the noisy drawing crop worse. Tags only beat the baseline after the dual read in E2.
  - Without multi-rotation OCR, dimension F1 is ~0.01. Tesseract does not read 30°-rotated text at all. This single change is the largest jump.
  - Unpreprocessed scanned pages with `psm 11` can take **minutes** per page in Tesseract. Needed a timeout (ADR-002).
  - The validator catches only 29–36% of wrong drawings. Most residual errors are a missed tag or dimension, which no rule can see.
    It never flags a fully-correct drawing (0% false-alarm), so `valid=true` is not a guarantee, but `needs_review` is a reliable signal.
- **Failures.** E2 errors by field (count of drawings): dimensions 17, tags 11, BOM 10, drawing_number 9, revision 7, design_temp 6,
  design_pressure 5, line_number 3, size 3, service 3, spec 1, material 1. Almost all title-block errors are on scanned drawings.
- **Next.** Count expected dimensions from the detected pipe segments (Hough on the drawing area) so a missing dimension becomes a validation failure.

## EXP-003 — Tag rules — dev

- **Hypothesis.** Tag misses come from (1) prefix misreads (`UT-1541` for `LT-1541`), (2) the dash splitting the token (`PSV` `-` `988`), (3) binarization losing thin strokes.
- **Change.** Fuzzy prefix snapping to the tag-prefix list, merge of up to 3 adjacent words on one OCR line, a second 0° read on the raw grayscale crop, de-duplicated by box IoU keeping the most confident read.
- **Result (dev, n=30).** Tags F1 0.876 → 0.936; drawing EM 0.233 → 0.400. Dims/BOM unchanged.
- **Failure noted.** Some tags are unreadable in both reads (`|-3964` for `PI-3964`); the prefix is lost entirely and no rule can recover it.

## EXP-002 — BOM catalog snapping + size-suffix repair — dev

- **Hypothesis.** BOM rows fail on small OCR slips (`21N` for `2IN`, `SMLS BE` with `PIPE` dropped, `I3IE` for `BE`).
- **Change.** Snap description stem to the material catalog (difflib ratio ≥ 0.6), unit from the catalog, size token rewritten from the validated nominal size.
- **Result (dev).** BOM F1 0.835 → 0.972; drawing EM 0.100 → 0.233.
- **Caveat.** Catalog = generator vocabulary. On real drawings the catalog would be much larger and snapping less decisive.

## EXP-001 — First end-to-end ablation A–E — dev (seed 1, n=30)

- Field EM: A 0.007 → B 0.563 → C 0.889 → D 0.889 → E 0.944. Drawing EM: 0 / 0 / 0 / 0.067 / 0.100. Dims F1 0 → 0.964 at D.
- **Bugs found on the way (fixed before this run):**
  - Table rows: the vertical-rule kernel (`h//8`) erased the vertical strokes of `E` in upscaled row bands → `GATIE VALVIE`. Kernel now 0.8 × crop height.
  - Scanned pages are skewed up to 1.2°, so horizontal rules span several pixel rows and row splitting failed → added Hough-based deskew.
  - Generator: tags and dimension text were overlapping pipe runs; added collision checks against pipe geometry and stopped leaders short of labels.
  - OpenCV 5 returns `HoughLinesP` lines in a different shape than 4.x; reshaped defensively.

## Resume-claim comparison

> Resume claim: 95%+ exact match | Reproduced here on synthetic isometrics (generator seed 2, n=60, 40% scanned): 93.0% field-level exact match (100.0% on the 36 clean drawings, 82.4% on the 24 scanned); 53.3% drawing-level exact match (every field incl. all tags, dimensions and BOM rows) | Gap / why: the original metric's definition and data (real client drawings) are not available; this synthetic set is cleaner in layout but harsher in scan noise than typical vector-PDF isometrics, and drawing-level exact match is a much stricter metric than field-level.
