"""Stage 5: Vision-Language Model re-read of fields that failed validation.

The VLM is never trusted blindly: each answer is parsed by the same field
grammar as the OCR text and only accepted if the record then passes the
field's validation rules. Rejected answers are logged in _meta.vlm.

Providers
  none       - no VLM; failing fields are left in _meta.needs_review (default demo)
  smolvlm    - local open model HuggingFaceTB/SmolVLM-256M-Instruct (CPU, no API key)
  anthropic  - Anthropic Messages API; needs ANTHROPIC_API_KEY (model via ANTHROPIC_MODEL)
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np
from PIL import Image

from . import parse

FIELD_LABEL = {
    "drawing_number": "DWG NO",
    "line_number": "LINE NO",
    "nominal_size_in": "SIZE",
    "pipe_spec": "SPEC",
    "material": "MATERIAL",
    "design_pressure_barg": "DESIGN PRESS",
    "design_temp_c": "DESIGN TEMP",
    "revision": "REV",
}

_SMOL = {}


def _question(label: str) -> str:
    return (f"This is the title block of an engineering drawing. What value is written to the right of "
            f"the label '{label}:'? Answer with the value only, exactly as written.")


def _ask_smolvlm(img: Image.Image, q: str) -> str:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    mid = os.getenv("SMOLVLM_MODEL", "HuggingFaceTB/SmolVLM-256M-Instruct")
    if mid not in _SMOL:
        _SMOL[mid] = (AutoProcessor.from_pretrained(mid), AutoModelForImageTextToText.from_pretrained(mid, torch_dtype=torch.float32))
    proc, model = _SMOL[mid]
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(text=prompt, images=[img.convert("RGB")], return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
    txt = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return txt.strip()


def _ask_anthropic(img: Image.Image, q: str) -> str:
    import anthropic  # optional dependency

    buf = io.BytesIO()
    img.convert("L").save(buf, format="PNG")
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        max_tokens=50,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(buf.getvalue()).decode()}},
            {"type": "text", "text": q}]}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


def refine_fields(rec: dict, problems: set[str], title_crop: np.ndarray, provider: str, meta: dict) -> None:
    from .pipeline import validate  # local import to avoid a cycle

    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        meta.setdefault("vlm", []).append({"error": "ANTHROPIC_API_KEY not set; VLM stage skipped"})
        return
    ask = {"smolvlm": _ask_smolvlm, "anthropic": _ask_anthropic}[provider]
    img = Image.fromarray(title_crop)
    log = meta.setdefault("vlm", [])
    for f in sorted(problems):
        if f not in FIELD_LABEL:
            continue
        label = FIELD_LABEL[f]
        try:
            ans = ask(img, _question(label))
        except Exception as e:  # network/model failure must not kill the pipeline
            log.append({"field": f, "error": repr(e)[:200]})
            continue
        parsed = parse.parse_title_block([f"{label}: {ans}"]).get(f)
        old = rec.get(f)
        rec[f] = parsed
        accepted = parsed is not None and f not in validate(rec)
        if accepted:
            meta["provenance"][f] = f"vlm:{provider}"
        else:
            rec[f] = old
        log.append({"field": f, "answer": ans[:80], "parsed": parsed, "accepted": accepted})
