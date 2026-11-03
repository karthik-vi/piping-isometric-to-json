"""Smoke tests: generator -> pipeline -> evaluation. Needs the tesseract binary."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from iso2json.evaluate import aggregate, score_one
from iso2json.generator import GenConfig, generate
from iso2json.pipeline import PipelineConfig, extract, validate

ROOT = Path(__file__).resolve().parents[1]


def test_generator_is_deterministic():
    a, ga = generate(GenConfig(seed=5))
    b, gb = generate(GenConfig(seed=5))
    assert ga == gb
    assert a.tobytes() == b.tobytes()


def test_validate_flags_cross_field_conflict():
    _, gt = generate(GenConfig(seed=1))
    assert validate(gt) == set()
    bad = dict(gt, pipe_spec="Z9Z")
    assert "pipe_spec" in validate(bad)


def test_clean_sample_end_to_end():
    png = ROOT / "data" / "samples" / "iso_clean_0001.png"
    gt = json.loads(png.with_suffix(".json").read_text())
    pred = extract(Image.open(png), PipelineConfig())
    s = score_one(gt, pred)
    # title-block scalar fields on the clean sample should all be right
    assert s["scalar_exact"], {k: (gt[k], pred[k]) for k in gt if gt[k] != pred.get(k)}
    agg = aggregate([s])
    assert 0.0 <= agg["tags_f1"] <= 1.0


def test_cli_writes_json(tmp_path):
    png = ROOT / "data" / "samples" / "iso_clean_0001.png"
    out = tmp_path / "o.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), str(png), "-o", str(out)],
                       capture_output=True, text=True)
    assert r.returncode in (0, 2), r.stderr
    rec = json.loads(out.read_text())
    assert "_meta" in rec and "needs_review" in rec["_meta"]


@pytest.mark.parametrize("variant", ["A", "E2"])
def test_evaluate_script_runs(tmp_path, variant):
    data = tmp_path / "d"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_dataset.py"), "--out", str(data), "--n", "2", "--seed", "7"], check=True)
    out = tmp_path / "r.json"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--data", str(data), "--variants", variant,
                    "--out", str(out), "--workers", "2"], check=True)
    r = json.loads(out.read_text())
    assert r["variants"][variant]["n"] == 2
