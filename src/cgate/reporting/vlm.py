"""Optional VLM phrases over a 4-sequence mosaic -- advisory, never authoritative.

The VLM contributes *qualitative appearance* only: signal character, margins,
enhancement pattern, mass effect. Presence/absence of every structured finding
comes from the calibrated system and is stated by the template. The division
matters because the VLM is not calibrated and cannot be gate-assessed: nothing
here feeds the confidence or the gate decision, and a contradicting phrase is
vetoed by `grounding.filter_phrases` before it reaches the report.

Model: google/medgemma-4b-it by default (medical-domain multimodal, ~8 GB bf16,
fits a 12 GB card), falling back to Qwen/Qwen2-VL-2B-Instruct. If neither can
be loaded (no GPU memory, no network), `describe()` returns [] and reports
render deterministic-template-only -- the VLM is strictly additive.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# BraTS modality order in the slice cache.
MODALITY_LABELS = ("T1", "T1ce", "T2", "FLAIR")

PROMPT_TEMPLATE = """\
You are looking at one axial slice of a pre-operative brain MRI study, shown as \
a 2x2 panel: top-left {m0}, top-right {m1}, bottom-left {m2}, bottom-right {m3}.

A calibrated vision system has already produced these structured findings \
(treat them as authoritative; your description must not contradict them):
{findings}

In at most {max_phrases} short bullet points, describe the qualitative appearance \
of the lesion: signal characteristics on each sequence, margins, enhancement \
pattern, mass effect, and any resection cavity or midline shift.

Reply ONLY with a JSON array of strings, each one sentence. No prose outside the array.\
"""


@dataclass
class VLMConfig:
    model_id: str = "google/medgemma-4b-it"
    fallback_model_id: str | None = "Qwen/Qwen2-VL-2B-Instruct"
    device: str | None = None  # None -> cuda if available, else cpu
    max_new_tokens: int = 256
    max_phrases: int = 3
    slice_index: int | None = None  # None -> middle slice


# ---------------------------------------------------------------------------
# Mosaic construction
# ---------------------------------------------------------------------------


def _to_uint8(slice2d: np.ndarray) -> np.ndarray:
    """Percentile-clip normalisation to 8-bit per panel.

    Raw MRI intensities vary by orders of magnitude between sequences, so a
    shared global normalisation would render T2 and FLAIR either black or
    blown out next to T1. Each panel is normalised independently.
    """
    lo, hi = np.percentile(slice2d, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((slice2d - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def build_mosaic(case_slices: np.ndarray, slice_index: int | None = None):
    """(n_slices, n_modalities, H, W) -> labelled 2x2 PIL image."""
    from PIL import Image, ImageDraw

    n_slices = case_slices.shape[0]
    idx = n_slices // 2 if slice_index is None else int(slice_index) % n_slices
    panels = [_to_uint8(case_slices[idx, m]) for m in range(case_slices.shape[1])]

    h, w = panels[0].shape
    gap, label_h = 4, 22
    canvas = Image.new("RGB", (2 * w + gap, 2 * (h + label_h) + gap), "black")
    draw = ImageDraw.Draw(canvas)
    positions = ((0, 0), (w + gap, 0), (0, h + label_h + gap), (w + gap, h + label_h + gap))
    for panel, (px, py), label in zip(panels, positions, MODALITY_LABELS):
        canvas.paste(Image.fromarray(panel).convert("RGB"), (px, py + label_h))
        draw.text((px + 6, py + 4), label, fill="yellow")
    return canvas


def mosaic_for_patient(cache, patient_id: str, slice_index: int | None = None):
    """Look a patient up in an open SliceCache and build their mosaic."""
    row = int(cache.rows_for([patient_id])[0])
    return build_mosaic(cache.get(row), slice_index=slice_index)


# ---------------------------------------------------------------------------
# The reporter
# ---------------------------------------------------------------------------


@dataclass
class VLMReporter:
    config: VLMConfig = field(default_factory=VLMConfig)
    model: object = None
    processor: object = None
    loaded_model_id: str | None = None
    available: bool = True
    _error: str | None = None

    # -- loading ------------------------------------------------------------

    def _device(self) -> str:
        if self.config.device:
            return self.config.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # torch optional for template-only use
            return "cpu"

    def _dtype(self):
        try:
            import torch

            return torch.bfloat16 if self._device().startswith("cuda") else torch.float32
        except Exception:
            return None

    def _try_load(self, model_id: str) -> bool:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=self._dtype(),
                device_map=self._device(),
            )
            model.eval()
            self.processor, self.model, self.loaded_model_id = processor, model, model_id
            return True
        except Exception as exc:  # network, OOM, unsupported arch -- all degrade
            self._error = f"{type(exc).__name__}: {exc}"
            return False

    def ensure_loaded(self) -> bool:
        if self.model is not None:
            return True
        if not self.available:
            return False
        candidates = [self.config.model_id]
        if self.config.fallback_model_id:
            candidates.append(self.config.fallback_model_id)
        for mid in candidates:
            print(f"  loading VLM {mid} ...", flush=True)
            if self._try_load(mid):
                print(f"  VLM ready: {mid}")
                return True
        self.available = False
        print(f"  VLM unavailable ({self._error}); reports fall back to template-only")
        return False

    # -- generation ---------------------------------------------------------

    def describe(self, image, findings_context: str) -> list[str]:
        """Raw phrases for one mosaic. [] whenever the VLM cannot run."""
        if image is None or not self.ensure_loaded():
            return []
        try:
            prompt = PROMPT_TEMPLATE.format(
                m0=MODALITY_LABELS[0],
                m1=MODALITY_LABELS[1],
                m2=MODALITY_LABELS[2],
                m3=MODALITY_LABELS[3],
                findings=findings_context,
                max_phrases=self.config.max_phrases,
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            # The image rides in the message content, not as an `images=`
            # kwarg: in transformers >= 5 the chat template resolves images
            # from the content itself, and passing both raises "multiple
            # values for keyword argument 'images'".
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            with _no_grad():
                out = self.model.generate(
                    **inputs, max_new_tokens=self.config.max_new_tokens, do_sample=False
                )
            text = self.processor.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            return _parse_phrases(text, limit=self.config.max_phrases)
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            return []


def _no_grad():
    try:
        import torch

        return torch.no_grad()
    except Exception:
        import contextlib

        return contextlib.nullcontext()


def _flatten_items(items: list) -> list[str]:
    """Turn parsed JSON items into plain sentence strings.

    VLMs answer in two shapes: a list of strings, or a list of
    {"category": "observation"} objects. Stringifying the objects would print
    the dict repr into the report, so dict values (the actual prose) are
    extracted and the keys (category labels) dropped.
    """
    out: list[str] = []
    for item in items:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            values = [str(v) for v in item.values() if isinstance(v, (str, int, float))]
            if values:
                out.append(". ".join(v.rstrip(".") for v in values) + ".")
    return out


def _parse_phrases(text: str, limit: int = 3) -> list[str]:
    """Pull a JSON string array out of possibly-chatty model output."""
    text = text.strip()
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        try:
            items = json.loads(match.group(0))
            if isinstance(items, list):
                flat = _flatten_items(items)
                if flat:
                    return flat[:limit]
        except json.JSONDecodeError:
            pass
    # Fallback: bullet/numbered lines. Chat templates vary; the contract is the
    # phrase list, not the envelope it arrived in. Preferring explicitly
    # bulleted lines stops conversational openers ("Sure! Here you go:") from
    # being reported as observations.
    bulleted: list[str] = []
    bare: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        is_bullet = bool(re.match(r"^(?:[-*•]|\d+[.)])\s+", stripped))
        clean = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", stripped).strip()
        if len(clean) < 8:
            continue
        (bulleted if is_bullet else bare).append(clean)
        if len(bulleted) >= limit:
            break
    lines = bulleted or bare
    return lines[:limit]
