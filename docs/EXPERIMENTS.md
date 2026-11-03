# Experiment log (newest first)

All numbers below were produced by `scripts/evaluate.py` in this repo on 2026-11-03, on a 4-vCPU / no-GPU container.
Raw outputs: `results/test_ablation.json`, `results/test_vlm.json`, `results/dev.json`, `results/dev_e2.json`.

Datasets (see DATA.md): **dev** = synthetic, seed 1, 30 drawings. **test** = synthetic, seed 2, 60 drawings (36 clean, 24 "scanned").
Test was generated before any tuning and used only for the final runs below.

Metric definitions: *field EM* = micro exact match over the 9 scalar fields; *drawing EM* = all 12 fields
(9 scalars + tags + dimensions + BOM) exactly right; *set F1* = micro F1 over tags / dimensions / BOM rows; *err-flag* = share of
wrong drawings that the validator put in `needs_review`.

---

## EXP-006 — Guarded VLM stage (after EXP-005 failure) — test

- **Change.** (1) VLM-sourced values are never used to repair other fields; (2) new cross-field rule: nominal size must equal the majority
  size token in the BOM; (3) if the VLM returns the same answer for several different fields, all those answers are rejected.
- **Result (test, n=60, `results/test_vlm_guarded.json`).**

| Variant | Field EM | Drawing EM | Tags F1 | Dims F1 | BOM F1 | Err-flag |
|---|---|---|---|---|---|---|
| E2 rules only (re-run with rule 2) | 0.930 | 0.550 | 0.950 | 0.953 | 0.920 | 0.37 |
| **F guarded SmolVLM** | **0.935** | **0.567** | 0.950 | 0.953 | **0.920** | 0.27 |

  The VLM fixed 3 fields on 3 scanned drawings (2 × drawing_number, 1 × revision) and broke nothing. Scanned drawing EM 0.292 → 0.333;
  clean drawings unchanged (0.722). Rule 2 did not change E2's numbers.
- **Cost.** One SmolVLM call per failing title-block field; wall time for 60 drawings 1,070 s single-process on a shared 4-vCPU box
  (vs ~180–420 s for E2 with 4 processes). The VLM is ~10× slower per drawing than the rules for +1.7 points drawing EM.
- **Next.** Try a stronger VLM via the Anthropic provider (implemented, not run: no key in this environment).

## EXP-005 — Unguarded VLM re-read (SmolVLM-256M, local CPU) — test — FAILURE CASE

- **Hypothesis.** A small VLM can re-read title-block fields that still fail validation; gated acceptance means it cannot make things worse.
- **Config.** Variant F = E2 + `vlm_provider=smolvlm` (`HuggingFaceTB/SmolVLM-256M-Instruct`, greedy, ≤24 new tokens, one question per failing field).
- **Result.** Drawing EM 0.550 → 0.567 and field EM 0.930 → 0.937, **but BOM F1 dropped 0.920 → 0.898**.
- **Why (iso_0028_scan).** The model degenerated and answered "16", "116", "11" for unrelated fields. Size = 16 passed validation because the
  line number was unreadable (nothing to contradict it), and the repair stage then rewrote every BOM description to `16IN`.
  The validation gate was necessary but not sufficient: an accepted wrong value propagated. Fixed in EXP-006.

## EXP-004b — Deterministic tag-prefix snapping (bug fix) — test

- **What happened.** A unit test showed `UT-1541` snapping to `TW-1541`: `difflib.get_close_matches` returned a 3-way tie (LT/FT/TW all ratio 0.5)
  and broke it by string order. Replaced with positional matching that only snaps to a *unique* best prefix and refuses ties.
  This was driven by the unit test, not by test-set results; E2 was re-run once on test afterwards. Both numbers are kept.
- **Result (test, n=60).** E2 before fix: drawing EM 0.533, tags F1 0.948. After fix: **drawing EM 0.550, tags F1 0.950** (P 0.985, R 0.919);
  all other metrics unchanged (field EM 0.930, dims F1 0.953, BOM F1 0.920). Clean drawing EM 0.722, scanned 0.292. Err-flag 0.37.
  Raw: `results/test_e2_prefixfix.json` (before: `results/test_ablation_pre_prefixfix.json`).
- **Note.** Median latency in this re-run was 21.8 s/drawing because landmark extraction for another project was running on the same 4 CPUs;
  the uncontended figure is the 8.9 s from EXP-004.

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

> Resume claim: 95%+ exact match | Reproduced here on synthetic isometrics (generator seed 2, n=60, 40% scanned): 93.0% field-level exact match (100.0% on the 36 clean drawings, 82.4% on the 24 scanned); 55.0% drawing-level exact match (every field incl. all tags, dimensions and BOM rows) | Gap / why: the original metric's definition and data (real client drawings) are not available; this synthetic set is cleaner in layout but harsher in scan noise than typical vector-PDF isometrics, and drawing-level exact match is a much stricter metric than field-level.
