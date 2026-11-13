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


# ----------------------------- Seedream 5.5 ------------------------------ #
# Seedream 5.5 adds native RGBA output (background=transparent), better
# multi-turn editing, and layer decomposition into a base image + up to 99
# layers with rich-text OCR for text layers.
# Docs: https://ai.byteplus.com/ark/docs/seedream-5-5

SEEDREAM_55_ID = "dola-seedream-5-5-261028"

_55_1K = _1K[:3] + [("1K 16:9", 1424, 800), ("1K 9:16", 800, 1424)] + _1K[5:]
_55_15K = [("1.5K 1:1", 1536, 1536), ("1.5K 4:3", 1792, 1344), ("1.5K 3:4", 1344, 1792),
           ("1.5K 16:9", 2048, 1152), ("1.5K 9:16", 1152, 2048), ("1.5K 3:2", 1872, 1248),
           ("1.5K 2:3", 1248, 1872), ("1.5K 21:9", 2352, 1008)]
_55_2K = [("2K 1:1", 2048, 2048), ("2K 4:3", 2368, 1776), ("2K 3:4", 1776, 2368),
          ("2K 16:9", 2816, 1584), ("2K 9:16", 1584, 2816), ("2K 3:2", 2496, 1664),
          ("2K 2:3", 1664, 2496), ("2K 21:9", 3136, 1344)]
# "2K (aspect from prompt)" sends the tier only and lets the model pick the
# shape from the prompt, which is BytePlus's recommended way.
_55_TIERS = [("2K (aspect from prompt)", "2K"), ("1.5K (aspect from prompt)", "1.5K"),
             ("1K (aspect from prompt)", "1K")]
SPEC_55 = {"presets": _55_2K + _55_15K + _55_1K, "pixels": (921_600, 4_624_220)}


def _key():
    key = _resolve_api_key("")
    if not key:
        raise ValueError("No ModelArk API key. Add it in Settings (gear icon) > BytePlus > "
                         "API keys > ModelArk API key.")
    return key


def _call(body):
    r = _post(_key(), body)
    if r.status_code >= 400:
        raise RuntimeError(f"Seedream 5.5 request failed [{r.status_code}]: "
                           f"{_redact(r.text[:800])}")
    data = [d for d in (r.json().get("data") or []) if isinstance(d, dict)]
    errors = [d["error"] for d in data if d.get("error")]
    if errors:
        print(f"{_LOG} {len(errors)} image(s) failed: {errors}")
    data = [d for d in data if d.get("url")]
    if not data:
        raise RuntimeError(f"Seedream 5.5 returned no image. {errors or _redact(r.text[:500])}")
    return data


def _download_rgba(url):
    """URL -> PIL RGBA image (opaque images get a solid alpha)."""
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGBA")


def _to_image_and_mask(rgba):
    """PIL RGBA -> ([1,H,W,3] image, [1,H,W] mask). The mask follows ComfyUI's
    Load Image convention: 1 where the pixel is transparent (1 - alpha)."""
    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    return (torch.from_numpy(arr[..., :3].copy()).unsqueeze(0),
            torch.from_numpy(1.0 - arr[..., 3]).unsqueeze(0))


def _resize_like(t, h, w):
    if t.shape[1:3] == (h, w):
        return t
    if t.dim() == 4:
        return torch.nn.functional.interpolate(t.movedim(-1, 1), size=(h, w),
                                               mode="bilinear").movedim(1, -1)
    return torch.nn.functional.interpolate(t.unsqueeze(1), size=(h, w),
                                           mode="bilinear").squeeze(1)


class BytePlusSeedream55(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BytePlusSeedream55",
            display_name="BytePlus Seedream 5.5",
            category=CAT,
            description="Text-to-image and image editing with Seedream 5.5: consistent "
            "multi-turn edits, up to 10 reference images and native transparent "
            "backgrounds (RGBA).",
            search_aliases=["seedream", "byteplus", "text to image", "image edit", "t2i",
                            "transparent", "rgba", "sticker", "5.5"],
            inputs=[
                IO.String.Input("prompt", multiline=True, default="",
                                tooltip="Describe the image or the edit. Call reference images "
                                "Image 1, Image 2... For edits, say what to change and what "
                                "to keep."),
                IO.DynamicCombo.Input(
                    "size",
                    options=[IO.DynamicCombo.Option(label, []) for label, _ in _55_TIERS]
                    + [IO.DynamicCombo.Option(label, []) for label, _, _ in SPEC_55["presets"]]
                    + [IO.DynamicCombo.Option("custom", [
                        IO.Int.Input("width", default=2048, min=64, max=8192, step=16),
                        IO.Int.Input("height", default=2048, min=64, max=8192, step=16),
                    ])],
                    tooltip="'aspect from prompt' lets the model choose the shape from your "
                    "prompt (recommended). Custom: 0.92-4.62 MP, aspect 1:16 to 16:1."),
                IO.Combo.Input("background", options=["auto", "transparent", "opaque"],
                               default="auto",
                               tooltip="'transparent' returns an RGBA image (stickers, icons, "
                               "product cut-outs); its alpha comes out as the mask. For "
                               "multi-turn edits, 'opaque' keeps results most consistent."),
                IO.Autogrow.Input("images", template=IO.Autogrow.TemplateNames(
                    IO.Image.Input("image", tooltip="Optional reference image to edit or blend."),
                    names=[f"image_{i}" for i in range(1, 11)], min=0)),
                IO.Combo.Input("output_format", options=["png", "jpeg"], default="png",
                               advanced=True, optional=True,
                               tooltip="jpeg cannot be transparent."),
                IO.Boolean.Input("watermark", default=False, advanced=True, optional=True,
                                 tooltip="Adds an 'AI generated' mark bottom-right."),
            ],
            outputs=[IO.Image.Output(),
                     IO.Mask.Output("mask", tooltip="Transparent areas (1 - alpha), like "
                                    "Load Image. All zeros for opaque images.")],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        import time
        return time.time()

    @classmethod
    def execute(cls, prompt, size, background, images=None, output_format="png",
                watermark=False):
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Write a prompt.")
        if background == "transparent" and output_format == "jpeg":
            raise ValueError("A transparent background needs output_format = png.")
        tiers = dict(_55_TIERS)
        size_value = tiers.get(size["size"])
        if not size_value:
            w, h = _resolve_size(SPEC_55, size)
            size_value = f"{w}x{h}"
        refs = []
        for tensor in (images or {}).values():
            refs.extend(_single_image_data_url(hwc) for hwc in _iter_images(tensor))
        if len(refs) > 10:
            raise ValueError(f"At most 10 reference images; {len(refs)} connected.")
        body = {"model": SEEDREAM_55_ID, "prompt": prompt, "size": size_value,
                "background": background, "output_format": output_format,
                "response_format": "url", "watermark": bool(watermark)}
        if refs:
            body["image"] = refs if len(refs) > 1 else refs[0]
        data = _call(body)
        pairs = [_to_image_and_mask(_download_rgba(d["url"])) for d in data]
        h, w = pairs[0][0].shape[1:3]
        images_out = torch.cat([_resize_like(i, h, w) for i, _ in pairs], dim=0)
        masks_out = torch.cat([_resize_like(m, h, w) for _, m in pairs], dim=0)
        return IO.NodeOutput(images_out, masks_out)


class BytePlusSeedream55Layers(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BytePlusSeedream55Layers",
            display_name="BytePlus Seedream 5.5 Layers",
            category=CAT,
            description="Splits one image into a filled-in base image and up to 99 "
            "transparent layers (subjects, text, objects), with optional OCR of text "
            "layers (text, position, font, colour).",
            search_aliases=["seedream", "byteplus", "layers", "decompose", "psd",
                            "separate", "ocr", "5.5"],
            inputs=[
                IO.Image.Input("image", tooltip="The image to split (one image; png or jpeg, "
                               "at least 512x512)."),
                IO.String.Input("prompt", multiline=True,
                                default="Decompose the elements in the image into independent "
                                "layers.",
                                tooltip="Optional: say what to separate, e.g. 'each animal "
                                "into its own layer; keep the rest in the base image'."),
                IO.Int.Input("max_images", default=30, min=2, max=100,
                             tooltip="Most images to return, base image included (2-100). "
                             "Higher values give finer splits."),
                IO.Combo.Input("size", options=["auto", "1K", "1.5K", "2K"], default="auto",
                               tooltip="Resolution of the base image and layers. 'auto' "
                               "keeps the input size within 1K-2K."),
                IO.Boolean.Input("text_recognition", default=True,
                                 tooltip="Also read regular text layers: text, line positions "
                                 "and approximate style (font, colour, bold...)."),
                IO.Boolean.Input("watermark", default=False, advanced=True, optional=True),
            ],
            outputs=[
                IO.Image.Output("base", tooltip="The image with every layer removed and "
                                "the gaps filled in."),
                IO.Image.Output("layers", tooltip="Each layer placed on a canvas the size "
                                "of the base image, bottom to top (z_index order)."),
                IO.Mask.Output("layer_masks", tooltip="Transparent areas of each layer "
                               "(1 - alpha), in the same order."),
                IO.String.Output("layer_info", tooltip="JSON: z_index, name, description, "
                                 "bounding box and OCR text of every layer. The transparent "
                                 "PNGs are also saved to output/seedream55_layers/<time>/."),
                IO.String.Output("text", tooltip="All recognized text, layer by layer."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        import time
        return time.time()

    @classmethod
    def execute(cls, image, prompt, max_images, size, text_recognition, watermark=False):
        frames = list(_iter_images(image))
        if len(frames) != 1:
            raise ValueError("Connect a single image (layer decomposition takes one image).")
        hwc = frames[0]
        if hwc.shape[0] * hwc.shape[1] < 512 * 512:
            raise ValueError("The image must have at least 512x512 pixels in total.")
        body = {"model": SEEDREAM_55_ID,
                "prompt": (prompt or "").strip() or "Decompose the elements in the image "
                "into independent layers.",
                "image": _single_image_data_url(hwc),
                "layer_decomposition": True,
                "layer_decomposition_options": {"max_images": int(max_images),
                                                "text_recognition": bool(text_recognition)},
                "size": size, "output_format": "png", "response_format": "url",
                "watermark": bool(watermark)}
        data = sorted(_call(body), key=lambda d: d.get("z_index", 0))
        base_item = next((d for d in data if d.get("z_index", 0) == 0), data[0])
        base_rgba = _download_rgba(base_item["url"])
        W, H = base_rgba.size
        base, _ = _to_image_and_mask(base_rgba.convert("RGB").convert("RGBA"))

        # ComfyUI's Save Image writes RGB, so the real transparent PNGs (ready
        # for Photoshop / Figma) are also saved here, with the layer info.
        import os
        import re
        import time
        from .nodes import OUTPUT_DIR
        folder = os.path.join(OUTPUT_DIR, "seedream55_layers", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(folder, exist_ok=True)
        base_rgba.save(os.path.join(folder, "00_base.png"))

        layers, masks, info, texts = [], [], [], []
        for d in data:
            if d is base_item:
                continue
            layer = _download_rgba(d["url"])
            # Restore the layer to its place on the base image: scale it to its
            # bounding box (absolute pixels) and paste it on a clear canvas.
            box = ((d.get("bounding_box") or {}).get("absolute")) or [0, 0, W, H]
            left, top, right, bottom = [int(v) for v in box]
            bw, bh = max(1, right - left), max(1, bottom - top)
            canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            canvas.alpha_composite(layer.resize((bw, bh), Image.LANCZOS),
                                   (max(0, left), max(0, top)))
            slug = re.sub(r"[^A-Za-z0-9]+", "_", (d.get("name") or "layer"))[:40].strip("_")
            canvas.save(os.path.join(folder, f"{int(d.get('z_index', 0)):02d}_{slug or 'layer'}.png"))
            img_t, mask_t = _to_image_and_mask(canvas)
            layers.append(img_t)
            masks.append(mask_t)
            entry = {k: d.get(k) for k in ("z_index", "name", "description", "bounding_box")}
            ocr = d.get("text_recognition")
            if ocr:
                entry["text_recognition"] = ocr
                texts.append(f"[layer {d.get('z_index')} {d.get('name') or ''}]\n"
                             f"{ocr.get('text', '')}")
            info.append(entry)

        if not layers:
            raise RuntimeError("Seedream 5.5 returned no layers. Try a different prompt "
                               "or a higher max_images.")
        import json
        layer_info = json.dumps({"size": [W, H], "folder": folder, "layers": info},
                                ensure_ascii=False, indent=2)
        with open(os.path.join(folder, "layers.json"), "w", encoding="utf-8") as fh:
            fh.write(layer_info)
        print(f"{_LOG} Seedream 5.5 Layers: base {W}x{H} + {len(layers)} layer(s)"
              + (f", {len(texts)} with OCR" if texts else "")
              + f". Transparent PNGs saved to {folder}")
        return IO.NodeOutput(base, torch.cat(layers, dim=0), torch.cat(masks, dim=0),
                             layer_info, "\n\n".join(texts))


NODES = [BytePlusSeedream55, BytePlusSeedream55Layers,
         BytePlusSeedream5Pro, BytePlusSeedream5Lite]
# Async, so independent BytePlus nodes in one workflow run at the same time.
NODES = [run_in_parallel(n) for n in NODES]
NODE_CLASS_MAPPINGS = {n.define_schema().node_id: n for n in NODES}
NODE_DISPLAY_NAME_MAPPINGS = {n.define_schema().node_id: n.define_schema().display_name
                              for n in NODES}
