# Deployment

## Local

```bash
sudo apt-get install -y tesseract-ocr fonts-dejavu-core   # macOS: brew install tesseract
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py data/samples/iso_clean_0001.png -o out/iso_clean_0001.json
pytest -q
```
Exit code is `0` if the JSON passed validation, `2` if some fields are in `_meta.needs_review`.

Optional VLM stage:
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
pip install -r requirements-vlm.txt
python scripts/run_pipeline.py data/samples/iso_0000_scan.png --vlm smolvlm      # local, no key, ~0.5 GB download
ANTHROPIC_API_KEY=... python scripts/run_pipeline.py drawing.png --vlm anthropic  # untested in this repo (no key available)
```

## Docker

```bash
docker build -t iso2json .
docker run --rm iso2json                                   # runs on the bundled clean sample
docker run --rm -v "$PWD/my:/data" iso2json python scripts/run_pipeline.py /data/drawing.png
```
The image does not include torch/transformers (keeps it ~400 MB). Add `requirements-vlm.txt` if you need the VLM stage.

## Cost / latency (measured on this container: 4 vCPU, no GPU)

- Rules-only pipeline (variant E2): see `docs/EXPERIMENTS.md` for the measured median latency per drawing.
  Most time is Tesseract: 5 rotation passes over the drawing area + 1 raw pass + ~17 row crops.
- SmolVLM-256M on CPU: several seconds per field query; only called for fields that fail validation.
- Anthropic provider: one API call per failing field (≤ 8 per drawing). Cost not measured (no key).

## Not production-ready

- No service/API wrapper, no queue, no auth. It is a batch CLI.
- No confidence calibration: `needs_review` catches only errors that break a grammar or cross-field rule.
  Tag / dimension misses are mostly **not** flagged (see EXPERIMENTS: error-flag rate).
- Layout detection assumes one sheet template (frame left, BOM top-right, title block bottom-right),
  with a fixed-coordinate fallback.
- Not validated on any real drawing.
