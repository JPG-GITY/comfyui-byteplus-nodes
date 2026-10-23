"""
Simple Seedream 5.0 image nodes (Pro and Lite), shaped like ComfyUI's partner
Seedream node: prompt, size, optional reference images, image out.

  POST {ark}/images/generations  -> data[].url  (links valid 24 h)

Auth is the same ModelArk key as Seedance (BYTEPLUS_API_KEY in .env).
Reference images go inline as base64 PNG data URLs (no TOS needed).

DISCLAIMER - PROOF-OF-CONCEPT DEMONSTRATION: see nodes.py; the same terms apply.
"""

import io

import numpy as np
import requests
import torch
from PIL import Image
from comfy_api.latest import io as IO

from .nodes import (
    DEFAULT_BASE_URL,
    _check_interrupt,
    _iter_images,
    _redact,
    _resolve_api_key,
    _single_image_data_url,
    run_in_parallel,
)

CAT = "BytePlus/Seedream"
_LOG = "[BytePlusSeedream]"

# Model id per node. Pro is published as dola-seedream-5-0-pro on the direct
# BytePlus route and as seedream-5-0-pro behind some proxies, so a 404 on the
# first id retries the second.
PRO_IDS = ("dola-seedream-5-0-pro-260628", "seedream-5-0-pro-260628")
LITE_IDS = ("seedream-5-0-260128", "seedream-5-0-lite-260128")

_1K = [("1K 1:1", 1024, 1024), ("1K 3:4", 864, 1152), ("1K 4:3", 1152, 864),
       ("1K 16:9", 1312, 736), ("1K 9:16", 736, 1312), ("1K 2:3", 832, 1248),
       ("1K 3:2", 1248, 832), ("1K 21:9", 1568, 672)]
_2K = [("2K 1:1", 2048, 2048), ("2K 3:4", 1728, 2304), ("2K 4:3", 2304, 1728),
       ("2K 16:9", 2848, 1600), ("2K 9:16", 1600, 2848), ("2K 2:3", 1664, 2496),
       ("2K 3:2", 2496, 1664), ("2K 21:9", 3136, 1344)]
_3K = [("3K 1:1", 3072, 3072), ("3K 3:4", 2592, 3456), ("3K 4:3", 3456, 2592),
       ("3K 16:9", 4096, 2304), ("3K 9:16", 2304, 4096), ("3K 2:3", 2496, 3744),
       ("3K 3:2", 3744, 2496), ("3K 21:9", 4704, 2016)]
_4K = [("4K 1:1", 4096, 4096), ("4K 3:4", 3520, 4704), ("4K 4:3", 4704, 3520),
       ("4K 16:9", 5504, 3040), ("4K 9:16", 3040, 5504), ("4K 2:3", 3328, 4992),
       ("4K 3:2", 4992, 3328), ("4K 21:9", 6240, 2656)]

# Per model: size presets, total-pixel range, max reference images, and
# whether it can return a set of related images (sequential generation).
SPECS = {
    "pro": {"ids": PRO_IDS, "presets": _1K + _2K, "default": "2K 1:1",
            "pixels": (921_600, 4_624_220), "max_refs": 10, "batch": False,
            "png": False},
    "lite": {"ids": LITE_IDS, "presets": _2K + _3K + _4K, "default": "2K 1:1",
             "pixels": (3_686_400, 16_777_216), "max_refs": 14, "batch": True,
             "png": True},
}


def _size_input(spec):
    lo, hi = spec["pixels"]
    return IO.DynamicCombo.Input(
        "size",
        options=[IO.DynamicCombo.Option(label, []) for label, _, _ in spec["presets"]]
        + [IO.DynamicCombo.Option("custom", [
            IO.Int.Input("width", default=2048, min=64, max=8192, step=16),
            IO.Int.Input("height", default=2048, min=64, max=8192, step=16),
        ])],
        tooltip=f"Output size. Custom sizes must total {lo / 1e6:.2f}-{hi / 1e6:.2f} MP "
        "with an aspect ratio between 1:16 and 16:1.",
    )


def _resolve_size(spec, size):
    choice = size["size"]
    for label, w, h in spec["presets"]:
        if label == choice:
            return w, h
    w, h = int(size["width"]), int(size["height"])
    lo, hi = spec["pixels"]
    if not lo <= w * h <= hi:
        raise ValueError(f"{w}x{h} is {w * h / 1e6:.2f} MP; this model needs "
                         f"{lo / 1e6:.2f}-{hi / 1e6:.2f} MP.")
    if not 1 / 16 <= w / h <= 16:
        raise ValueError(f"{w}x{h}: the aspect ratio must be between 1:16 and 16:1.")
    return w, h


def _post(key, body):
    _check_interrupt()
    r = requests.post(f"{DEFAULT_BASE_URL}/images/generations",
                      headers={"Authorization": f"Bearer {key}",
                               "Content-Type": "application/json"},
                      json=body, timeout=600)
    return r


def _download_image(url):
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    im = Image.open(io.BytesIO(r.content)).convert("RGB")
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).unsqueeze(0)


def _generate(spec, prompt, size, images=None, count=1, watermark=False):
    key = _resolve_api_key("")
    if not key:
        raise ValueError("No ModelArk API key. Add it in Settings (gear icon) > BytePlus > "
                         "API keys > ModelArk API key.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Write a prompt.")
    w, h = _resolve_size(spec, size)

    refs = []
    for tensor in (images or {}).values():
        refs.extend(_single_image_data_url(hwc) for hwc in _iter_images(tensor))
    if len(refs) > spec["max_refs"]:
        raise ValueError(f"At most {spec['max_refs']} reference images; {len(refs)} connected.")
    if count > 1 and len(refs) + count > 15:
        raise ValueError("Reference images plus generated images cannot exceed 15.")

    body = {"prompt": prompt, "size": f"{w}x{h}", "response_format": "url",
            "watermark": bool(watermark)}
    if refs:
        body["image"] = refs if len(refs) > 1 else refs[0]
    if spec["batch"]:
        body["sequential_image_generation"] = "auto" if count > 1 else "disabled"
        if count > 1:
            body["sequential_image_generation_options"] = {"max_images": int(count)}
    if spec["png"]:
        body["output_format"] = "png"

    r = None
    for model_id in spec["ids"]:
        body["model"] = model_id
        r = _post(key, body)
        if r.status_code == 404 and "NotFound" in r.text:
            print(f"{_LOG} {model_id} not found on this account; trying the next id...")
            continue
        break
    if r.status_code >= 400:
        raise RuntimeError(f"Seedream request failed [{r.status_code}]: {_redact(r.text[:800])}")
    data = r.json().get("data") or []
    urls = [d["url"] for d in data if isinstance(d, dict) and d.get("url")]
    errors = [d.get("error") for d in data if isinstance(d, dict) and d.get("error")]
    if errors:
        print(f"{_LOG} {len(errors)} image(s) failed: {errors}")
    if not urls:
        raise RuntimeError(f"Seedream returned no image. {errors or _redact(r.text[:500])}")
    out = [_download_image(u) for u in urls]
    first = out[0]
    out = [t if t.shape == first.shape else torch.nn.functional.interpolate(
        t.movedim(-1, 1), size=first.shape[1:3], mode="bilinear").movedim(1, -1) for t in out]
    return IO.NodeOutput(torch.cat(out, dim=0))


def _schema(node_id, name, spec, description):
    inputs = [
        IO.String.Input("prompt", multiline=True, default="",
                        tooltip="Describe the image, or the edit to make to the reference "
                        "images (call them Image 1, Image 2...)."),
        _size_input(spec),
        IO.Autogrow.Input("images", template=IO.Autogrow.TemplateNames(
            IO.Image.Input("image", tooltip="Optional reference image to edit or blend."),
            names=[f"image_{i}" for i in range(1, spec["max_refs"] + 1)], min=0)),
    ]
    if spec["batch"]:
        inputs.append(IO.Int.Input("count", default=1, min=1, max=15,
                                   tooltip="Above 1, the model may return a set of related "
                                   "images (storyboard, variations)."))
    inputs.append(IO.Boolean.Input("watermark", default=False, advanced=True, optional=True,
                                   tooltip="Adds an 'AI generated' mark bottom-right."))
    return IO.Schema(
        node_id=node_id, display_name=name, category=CAT, description=description,
        search_aliases=["seedream", "byteplus", "text to image", "image edit", "t2i"],
        inputs=inputs, outputs=[IO.Image.Output()],
    )


class BytePlusSeedream5Pro(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema("BytePlusSeedream5Pro", "BytePlus Seedream 5.0 Pro", SPECS["pro"],
                       "Highest-quality text-to-image and image editing (1K-2K), with up "
                       "to 10 reference images.")

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        import time
        return time.time()

    @classmethod
    def execute(cls, prompt, size, images=None, watermark=False):
        return _generate(SPECS["pro"], prompt, size, images, 1, watermark)


class BytePlusSeedream5Lite(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _schema("BytePlusSeedream5Lite", "BytePlus Seedream 5.0 Lite", SPECS["lite"],
                       "Fast, lower-cost text-to-image and image editing at 2K-4K, with up "
                       "to 14 reference images and image sets.")

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        import time
        return time.time()

    @classmethod
    def execute(cls, prompt, size, count=1, images=None, watermark=False):
        return _generate(SPECS["lite"], prompt, size, images, count, watermark)


NODES = [BytePlusSeedream5Pro, BytePlusSeedream5Lite]
# Async, so independent BytePlus nodes in one workflow run at the same time.
NODES = [run_in_parallel(n) for n in NODES]
NODE_CLASS_MAPPINGS = {n.define_schema().node_id: n for n in NODES}
NODE_DISPLAY_NAME_MAPPINGS = {n.define_schema().node_id: n.define_schema().display_name
                              for n in NODES}
