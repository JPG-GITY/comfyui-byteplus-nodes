"""
ComfyUI nodes for BytePlus AI MediaKit (BytePlus VOD).

DISCLAIMER - PROOF-OF-CONCEPT DEMONSTRATION: see nodes.py; the same terms apply.

AI MediaKit exposes every tool behind one REST shape, so a single node covers
all of them. The tool list, each tool's parameters and a working request
template come from mediakit_tools.json, which tools/build_mediakit_catalog.py
regenerates from the official API reference.

Flow:
  async tools: POST {base}/tools/<tool>       -> {task_id}
               GET  {base}/tasks/{task_id}    (poll) -> status + result
  sync tools : POST {base}/tools-sync/<tool>  -> result in the response
  Result URLs expire after 24 h, so every file in the result is downloaded.

Docs:
  Overview : https://docs.byteplus.com/en/docs/byteplus-vod/ai-mediakit-overview
  Concepts : https://docs.byteplus.com/en/docs/byteplus-vod/ai-mediakit-basic-concepts-and-prerequisites
  Task     : https://docs.byteplus.com/en/docs/byteplus-vod/ai-mediakit-get-task-details
  Errors   : https://docs.byteplus.com/en/docs/byteplus-vod/ai-mediakit-error-codes

Auth: an AI MediaKit API key (Authorization: Bearer). This is NOT the ModelArk
ark- key: create it in the AI MediaKit console > Settings > API key, or use a
general-purpose IAM API key.

Inputs must be public http(s) URLs. Connected VIDEO / IMAGE / AUDIO sockets are
uploaded to your TOS bucket (same .env keys as the Seedance node) and passed as
presigned URLs.
"""

import os
import re
import io
import json
import time
import uuid
import shutil
import subprocess
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image

from .nodes import (
    OUTPUT_DIR,
    VideoFromFile,
    _HAS_VIDEO,
    _check_interrupt,
    _encode_audio,
    _env_lookup,
    _interruptible_sleep,
    _iter_images,
    _save_image_png,
    _tos_delete,
    _tos_upload,
)

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_CATALOG_PATH = os.path.join(_NODE_DIR, "mediakit_tools.json")

DEFAULT_MEDIAKIT_BASE_URL = "https://mediakit.ap-southeast-1.bytepluses.com/api/v1"
MEDIAKIT_CONSOLE_URL = ("https://console.byteplus.com/vodpaas/region:vodpaas+ap-southeast-1"
                        "/ai-mediakit/settings")

# Env vars checked for the MediaKit key (field blank -> look these up).
_MEDIAKIT_KEY_ENV = ("BYTEPLUS_MEDIAKIT_API_KEY", "MEDIAKIT_API_KEY",
                     "BYTEPLUS_VOD_MEDIAKIT_API_KEY")

# Sockets. Each kind has three, addressed in params_json as {video}, {video_2},
# {video_3} (same for image / audio). An IMAGE socket that carries a batch
# yields one URL per frame: {image} is the first, {image[2]} the second, ...
_KINDS = ("video", "image", "audio")
_SLOTS = {k: [k, f"{k}_2", f"{k}_3"] for k in _KINDS}

_VIDEO_EXT = (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".flv", ".ts", ".avi", ".wmv", ".m3u8")
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".avif")
_AUDIO_EXT = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma")
_OTHER_EXT = (".gif", ".srt", ".vtt", ".ass", ".json", ".txt", ".zip")

_TERMINAL_OK = {"completed", "succeeded", "success"}
_TERMINAL_BAD = {"failed", "canceled", "cancelled", "expired"}

_LOG = "[BytePlusMediaKit]"


# ----------------------------- catalog --------------------------------- #

_CATALOG = None


def _catalog():
    global _CATALOG
    if _CATALOG is None:
        with open(_CATALOG_PATH, "r", encoding="utf-8") as fh:
            _CATALOG = json.load(fh)
        for tool_id, spec in _CATALOG["tools"].items():
            spec["template"] = _make_template(spec)
    return _CATALOG


def _tool_label(tool_id, spec):
    return f"[{spec['category']}] {tool_id}"


def _tool_labels():
    return [_tool_label(t, s) for t, s in _catalog()["tools"].items()]


def _tool_from_label(label):
    """'[Video AI] enhance-video' -> 'enhance-video'. A bare id also works."""
    tool_id = label.split("]", 1)[-1].strip()
    if tool_id not in _catalog()["tools"]:
        raise ValueError(f"Unknown AI MediaKit tool '{label}'.")
    return tool_id


def _kind_for(key, url):
    """Guess what a URL holds from its extension, then from its field name."""
    path = urlparse(url).path.lower()
    if path.endswith(_VIDEO_EXT):
        return "video"
    if path.endswith(_IMAGE_EXT):
        return "image"
    if path.endswith(_AUDIO_EXT):
        return "audio"
    if path.endswith(_OTHER_EXT):
        return None
    key = (key or "").lower()
    if "video" in key or key == "source":
        return "video"
    if any(w in key for w in ("image", "frame", "cover", "logo", "mask", "watermark")):
        return "image"
    if any(w in key for w in ("audio", "voice", "music", "bgm", "speech")):
        return "audio"
    return None


def _make_template(spec):
    """The doc's first request example, with example.com media URLs swapped
    for socket placeholders ({video}, {video_2}, {image}, ...), and the
    optional bookkeeping fields left out."""
    examples = spec.get("examples") or []
    body = json.loads(json.dumps(examples[0])) if examples else {}
    counters = {k: 0 for k in _KINDS}

    def swap(value, key):
        if isinstance(value, dict):
            return {k: swap(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [swap(v, key) for v in value]
        if isinstance(value, str) and re.match(r"https?://[^/]*example\.", value):
            kind = _kind_for(key, value)
            if kind:
                counters[kind] += 1
                n = counters[kind]
                return "{" + (kind if n == 1 else f"{kind}_{n}") + "}"
        return value

    body = swap(body, "")
    for k in ("callback_url", "callback_args", "client_token", "queue_id"):
        if isinstance(body, dict):
            body.pop(k, None)
    return body


# ----------------------------- helpers --------------------------------- #

def _resolve_key(field_value=""):
    if field_value and field_value.strip():
        return field_value.strip()
    return _env_lookup(_MEDIAKIT_KEY_ENV)


def _base_url():
    return (_env_lookup(("MEDIAKIT_BASE_URL",)) or DEFAULT_MEDIAKIT_BASE_URL).rstrip("/")


def _redact(text):
    if not text:
        return text
    return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1***REDACTED***", str(text))


def _describe_error(data, status=None):
    err = (data or {}).get("error") if isinstance(data, dict) else None
    err = err if isinstance(err, dict) else {}
    code = err.get("code") or (f"HTTP {status}" if status else "Error")
    msg = err.get("message") or ""
    param = f" (param: {err['param']})" if err.get("param") else ""
    rid = (data or {}).get("request_id") if isinstance(data, dict) else None
    hint = ""
    if status == 401 or code in ("Unauthorized", "InvalidApiKey", "AuthenticationFailed"):
        hint = (" Check the AI MediaKit API key (AI MediaKit console > Settings > "
                f"API key: {MEDIAKIT_CONSOLE_URL}). A ModelArk ark- key does not work here.")
    elif status == 403 or code == "AccessDenied":
        hint = (" Your account may not have this tool enabled, or it needs allowlist "
                "access; check the AI MediaKit console or open a support ticket.")
    return _redact(f"{code}{param}: {msg}" + (f" Request ID: {rid}." if rid else "") + hint)


def _request(method, url, key, body=None, timeout=60, retries=3):
    """One MediaKit call; retries network errors, 429 and 5xx."""
    headers = {"Authorization": f"Bearer {key}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    last = None
    for attempt in range(1, retries + 1):
        _check_interrupt()
        try:
            r = requests.request(method, url, headers=headers, json=body, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            last = f"network error: {exc}"
        else:
            try:
                data = r.json()
            except ValueError:
                data = None
            if r.status_code < 400 and isinstance(data, dict):
                return data
            if r.status_code == 429 or r.status_code >= 500:
                last = _describe_error(data, r.status_code) if data else f"HTTP {r.status_code}"
            else:
                raise RuntimeError(f"AI MediaKit request failed: {_describe_error(data, r.status_code)}"
                                   if data else f"AI MediaKit returned HTTP {r.status_code}: "
                                   f"{_redact(r.text[:500])}")
        if attempt < retries:
            print(f"{_LOG} {last}; retry {attempt}/{retries - 1}...")
            _interruptible_sleep(2 ** attempt)
    raise RuntimeError(f"AI MediaKit request failed: {last}")


def _fill_placeholders(value, urls):
    """Replace {video}, {image_2}, {image[3]}, ... in every string."""
    pattern = re.compile(r"\{((?:video|image|audio)(?:_\d+)?)(?:\[(\d+)\])?\}")

    def resolve(m):
        slot, idx = m.group(1), int(m.group(2) or 1)
        found = urls.get(slot) or []
        if idx > len(found):
            raise ValueError(
                f"params_json uses {m.group(0)} but the '{slot}' socket "
                + ("is not connected." if not found else f"only has {len(found)} item(s).")
                + " Connect it, or replace the placeholder with a public http(s) URL.")
        return found[idx - 1]

    if isinstance(value, dict):
        return {k: _fill_placeholders(v, urls) for k, v in value.items()}
    if isinstance(value, list):
        out = []
        for v in value:
            # "{image[*]}" as a list item expands to every frame of the socket.
            if isinstance(v, str) and re.fullmatch(r"\{(video|image|audio)(_\d+)?\[\*\]\}", v):
                slot = v[1:-4]
                if not urls.get(slot):
                    raise ValueError(f"params_json uses {v} but '{slot}' is not connected.")
                out.extend(urls[slot])
            else:
                out.append(_fill_placeholders(v, urls))
        return out
    if isinstance(value, str):
        whole = pattern.fullmatch(value)
        if whole:
            return resolve(whole)
        return pattern.sub(resolve, value)
    return value


def _auto_fill(tool_spec, body, urls):
    """Fill the tool's primary URL fields from connected sockets when the
    params leave them out. Explicit values (and placeholders) always win."""
    names = {p["name"] for p in tool_spec.get("params", [])}
    for kind in _KINDS:
        everything = [u for slot in _SLOTS[kind] for u in urls.get(slot, [])]
        if not everything:
            continue
        if f"{kind}_url" in names and not body.get(f"{kind}_url"):
            body[f"{kind}_url"] = everything[0]
        if f"{kind}_urls" in names and not body.get(f"{kind}_urls"):
            body[f"{kind}_urls"] = everything
    return body


def _collect_urls(obj, key="", out=None):
    """Every http(s) URL in a result, with the field name it came from."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _collect_urls(v, k, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_urls(v, key, out)
    elif isinstance(obj, str) and obj.startswith(("http://", "https://")):
        if not any(u == obj for _, u in out):
            out.append((key, obj))
    return out


_MIME_EXT = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif",
    "image/bmp": ".bmp", "image/tiff": ".tif", "image/heic": ".heic", "image/avif": ".avif",
    "video/mp4": ".mp4", "video/quicktime": ".mov", "video/webm": ".webm",
    "video/x-matroska": ".mkv", "video/x-flv": ".flv", "video/mp2t": ".ts",
    "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/aac": ".aac",
    "audio/mp4": ".m4a", "audio/flac": ".flac", "audio/ogg": ".ogg",
    "application/json": ".json", "text/plain": ".txt",
}


def _download(url, folder, index):
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        # MediaKit result links often have no extension, or a CDN one such as
        # ".image"; take it from the Content-Type so the saved file opens
        # normally (and the magic bytes below when that is generic too).
        if ext not in _VIDEO_EXT + _IMAGE_EXT + _AUDIO_EXT + _OTHER_EXT:
            mime = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            ext = _MIME_EXT.get(mime, ".bin")
        name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(path))[:80] or "file"
        if not name.lower().endswith(ext):
            name += ext
        dest = os.path.join(folder, f"{index:03d}_{name}")
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                _check_interrupt()
                fh.write(chunk)
    if ext == ".bin":
        real = _magic_ext(dest)
        if real:
            os.replace(dest, dest[:-4] + real)
            dest = dest[:-4] + real
    return dest


def _magic_ext(path):
    """Extension from a file's first bytes, for downloads with no usable type."""
    with open(path, "rb") as fh:
        head = fh.read(16)
    if head.startswith(b"\x89PNG"):
        return ".png"
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return ".webp"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return ".wav"
    if head.startswith(b"GIF8"):
        return ".gif"
    if head[4:8] == b"ftyp":
        return ".mp4"
    if head.startswith(b"\x1aE\xdf\xa3"):
        return ".webm"
    if head.startswith(b"ID3") or head[:2] in (b"\xff\xfb", b"\xff\xf3"):
        return ".mp3"
    if head.startswith(b"fLaC"):
        return ".flac"
    return None


def _sniff_kind(path, hint):
    """Extension first; an unknown extension falls back to the field hint."""
    ext = os.path.splitext(path)[1].lower()
    for kind, exts in (("video", _VIDEO_EXT), ("image", _IMAGE_EXT), ("audio", _AUDIO_EXT)):
        if ext in exts:
            return kind
    if ext not in _OTHER_EXT and hint:
        return hint
    return None


def _load_images(paths):
    """Image files -> one [B,H,W,3] float tensor. Frames that differ in size
    are resized to the first one, because an IMAGE batch is a single tensor."""
    import torch
    frames, size = [], None
    for p in paths:
        try:
            im = Image.open(p)
            im.seek(0)
            im = im.convert("RGB")
        except Exception as exc:
            print(f"{_LOG} Skipping unreadable image {p}: {exc}")
            continue
        if size is None:
            size = im.size
        elif im.size != size:
            im = im.resize(size, Image.LANCZOS)
        frames.append(np.asarray(im, dtype=np.float32) / 255.0)
    if not frames:
        return None
    return torch.from_numpy(np.stack(frames))


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _load_audio(path):
    """Audio file (or a video's soundtrack) -> ComfyUI AUDIO dict."""
    import torch
    import soundfile as sf
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        ffmpeg = _ffmpeg_exe()
        if not ffmpeg:
            print(f"{_LOG} Cannot decode {path}: no ffmpeg available.")
            return None
        wav = os.path.splitext(path)[0] + ".decoded.wav"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", path, "-vn", wav],
                       check=True)
        data, sr = sf.read(wav, dtype="float32", always_2d=True)
        os.remove(wav)
    waveform = torch.from_numpy(data.T.copy()).unsqueeze(0)  # [1, C, T]
    return {"waveform": waveform, "sample_rate": int(sr)}


def _empty_image():
    import torch
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _empty_audio():
    import torch
    return {"waveform": torch.zeros((1, 1, 1), dtype=torch.float32), "sample_rate": 44100}


# --------------------------- task + outputs ----------------------------- #

def _poll(base, key, task_id, poll_interval, timeout_seconds, pbar=None):
    started = time.time()
    while True:
        _check_interrupt()
        data = _request("GET", f"{base}/tasks/{task_id}", key)
        status = str(data.get("status") or "").lower()
        if status in _TERMINAL_OK:
            return data
        if status in _TERMINAL_BAD:
            raise RuntimeError(f"AI MediaKit task {task_id} {status}: {_describe_error(data)}")
        elapsed = int(time.time() - started)
        if elapsed > timeout_seconds:
            raise RuntimeError(
                f"AI MediaKit task {task_id} still '{status or 'running'}' after "
                f"{timeout_seconds}s. It keeps running server-side: paste this task_id "
                "into resume_task_id on the 'BytePlus AI MediaKit (All Tools)' node to "
                "collect it later.")
        print(f"{_LOG} task {task_id}: {status or 'running'} ({elapsed}s)")
        if pbar is not None:
            pbar.update_absolute(min(elapsed, timeout_seconds - 1), timeout_seconds)
        _interruptible_sleep(poll_interval)


def save_results(task):
    """Download every file in a task result to output/mediakit/<task_id>/.

    Returns [{field, url, path, kind}] in result order; kind is video / image /
    audio / None. Links expire after 24 h, so this runs right away."""
    task_id = str(task.get("task_id") or uuid.uuid4().hex[:12])
    result = task.get("result") or task.get("output") or {}
    found = _collect_urls(result)
    files = []
    if not found:
        return files
    folder = os.path.join(OUTPUT_DIR, "mediakit", re.sub(r"[^A-Za-z0-9._-]", "_", task_id))
    os.makedirs(folder, exist_ok=True)
    for i, (field, url) in enumerate(found, 1):
        try:
            p = _download(url, folder, i)
        except Exception as exc:
            print(f"{_LOG} Could not download {field}: {_redact(exc)}")
            continue
        files.append({"field": field, "url": url, "path": p,
                      "kind": _sniff_kind(p, _kind_for(field, url))})
    print(f"{_LOG} Saved {len(files)} file(s) to {folder}")
    return files


def run_tool(tool_id, body, sockets=None, api_key="", timeout_seconds=3600,
             poll_interval=10, delete_tos_after_run=True):
    """Upload connected media, call one MediaKit tool and wait for the result.

    sockets maps slot names (video, video_2, image, audio, ...) to ComfyUI
    VIDEO / IMAGE / AUDIO values; placeholders in body ({video}, {image[*]}...)
    are filled from them, and the tool's primary *_url fields are filled
    automatically when left out. Returns the finished task dict."""
    key = _resolve_key(api_key)
    if not key:
        raise ValueError(
            "No AI MediaKit API key. Add it in Settings (gear icon) > BytePlus > API keys. "
            f"Create one at {MEDIAKIT_CONSOLE_URL}. The ModelArk ark- key does not work here.")
    spec = _catalog()["tools"][tool_id]
    sockets = sockets or {}
    uploaded, temp_files = [], []
    tmp_dir = os.path.join(OUTPUT_DIR, "mediakit", "_uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    stamp = int(time.time() * 1000)

    def upload(local):
        url, bucket, obj = _tos_upload(local)
        uploaded.append((bucket, obj))
        return url

    try:
        urls = {}
        for slot, value in sockets.items():
            if value is None:
                continue
            kind = slot.split("_")[0]
            if kind == "video":
                tmp = os.path.join(tmp_dir, f"mk_{stamp}_{slot}.mp4")
                value.save_to(tmp)
                temp_files.append(tmp)
                urls[slot] = [upload(tmp)]
            elif kind == "image":
                urls[slot] = []
                for i, hwc in enumerate(_iter_images(value), 1):
                    tmp = os.path.join(tmp_dir, f"mk_{stamp}_{slot}_{i}.png")
                    _save_image_png(hwc, tmp)
                    temp_files.append(tmp)
                    urls[slot].append(upload(tmp))
            elif kind == "audio":
                tmp = os.path.join(tmp_dir, f"mk_{stamp}_{slot}.wav")
                _encode_audio(value, tmp)
                temp_files.append(tmp)
                urls[slot] = [upload(tmp)]

        body = _fill_placeholders(body, urls)
        body = _auto_fill(spec, body, urls)
        missing = [p["name"] for p in spec.get("params", [])
                   if p.get("required") and p["name"] not in body]
        if missing:
            print(f"{_LOG} Warning: {tool_id} documents required field(s) "
                  f"{', '.join(missing)} that the request does not set.")

        base = _base_url()
        pbar = None
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(int(timeout_seconds))
        except Exception:
            pass

        print(f"{_LOG} {tool_id}: submitting ({'sync' if spec['sync'] else 'async'})")
        # Sync image tools answer with the result; give them long enough.
        data = _request("POST", base + spec["path"], key, body,
                        timeout=600 if spec["sync"] else 120,
                        retries=1 if spec["sync"] else 3)
        if data.get("success") is False:
            raise RuntimeError(f"AI MediaKit {tool_id} failed: {_describe_error(data)}")
        if spec["sync"] or data.get("result") is not None:
            return data
        task_id = data.get("task_id")
        if not task_id:
            raise RuntimeError(f"No task_id in AI MediaKit response: {_redact(data)}")
        print(f"{_LOG} {tool_id}: task {task_id} queued")
        return _poll(base, key, task_id, poll_interval, timeout_seconds, pbar)
    finally:
        if delete_tos_after_run and uploaded:
            _tos_delete(uploaded)
        for p in temp_files:
            try:
                os.remove(p)
            except OSError:
                pass


def resume_task(task_id, api_key="", timeout_seconds=3600, poll_interval=10):
    key = _resolve_key(api_key)
    if not key:
        raise ValueError("No AI MediaKit API key (BYTEPLUS_MEDIAKIT_API_KEY in .env).")
    return _poll(_base_url(), key, task_id.strip(), poll_interval, timeout_seconds)


def _build_outputs(task, download_results):
    """Download every file in the result and map it onto the node outputs."""
    task_id = str(task.get("task_id") or "")
    found = _collect_urls(task.get("result") or task.get("output") or {})
    files = save_results(task) if download_results else []
    paths = [f["path"] for f in files]
    by_kind = {k: [f["path"] for f in files if f["kind"] == k] for k in _KINDS}

    video = None
    if by_kind["video"] and _HAS_VIDEO:
        video = VideoFromFile(by_kind["video"][0])
    images = _load_images(by_kind["image"]) if by_kind["image"] else None
    audio = None
    if by_kind["audio"]:
        audio = _load_audio(by_kind["audio"][0])
    elif by_kind["video"]:
        try:
            audio = _load_audio(by_kind["video"][0])
        except Exception:
            audio = None

    url_lines = "\n".join(u for _, u in found)
    return (
        video,
        images if images is not None else _empty_image(),
        audio if audio is not None else _empty_audio(),
        json.dumps(task, ensure_ascii=False, indent=2),
        url_lines,
        "\n".join(paths),
        task_id,
    )


_RETURN_TYPES = ("VIDEO", "IMAGE", "AUDIO", "STRING", "STRING", "STRING", "STRING")
_RETURN_NAMES = ("video", "images", "audio", "result_json", "result_urls", "file_paths", "task_id")


# ------------------------------ nodes ---------------------------------- #

class BytePlusMediaKit:
    """Advanced: run any AI MediaKit tool (all 74) with a raw request body.
    The simple per-task nodes live in mediakit_nodes.py."""

    @classmethod
    def INPUT_TYPES(cls):
        labels = _tool_labels()
        first = _tool_from_label(labels[0])
        template = json.dumps(_catalog()["tools"][first]["template"], indent=2, ensure_ascii=False)
        optional = {}
        for kind, sock in (("video", "VIDEO"), ("image", "IMAGE"), ("audio", "AUDIO")):
            for slot in _SLOTS[kind]:
                optional[slot] = (sock, {"tooltip": f"Uploaded to TOS; use {{{slot}}} in params_json."})
        optional.update({
            "resume_task_id": ("STRING", {"default": "", "advanced": True, "tooltip":
                               "Collect an earlier async task instead of submitting a new "
                               "one (e.g. one that outlived timeout_seconds)."}),
            "api_key": ("STRING", {"default": "", "advanced": True, "tooltip":
                        "Leave blank to read BYTEPLUS_MEDIAKIT_API_KEY from .env. "
                        "Not the ModelArk ark- key."}),
            "timeout_seconds": ("INT", {"default": 3600, "min": 30, "max": 86400, "advanced": True,
                                        "tooltip": "How long to wait for an async task. "
                                        "Professional enhancement can take hours."}),
            "poll_interval": ("INT", {"default": 10, "min": 2, "max": 300, "advanced": True}),
            "download_results": ("BOOLEAN", {"default": True, "advanced": True,
                                             "tooltip": "Result links expire in 24 h; this "
                                             "saves every file to output/mediakit/<task_id>/."}),
            "delete_tos_after_run": ("BOOLEAN", {"default": True, "advanced": True}),
        })
        return {
            "required": {
                "tool": (labels, {"default": labels[0], "tooltip":
                         "Every AI MediaKit tool. Picking one loads its request template "
                         "into params_json."}),
                "params_json": ("STRING", {"multiline": True, "default": template, "tooltip":
                                "Request body sent to the tool, as in the API reference. "
                                "Placeholders {video} {video_2} {video_3} {image} {image_2} "
                                "{image_3} {audio} {audio_2} {audio_3} are replaced with the "
                                "connected sockets; {image[2]} picks a batch frame and "
                                "\"{image[*]}\" in a list expands to every frame."}),
            },
            "optional": optional,
        }

    RETURN_TYPES = _RETURN_TYPES
    RETURN_NAMES = _RETURN_NAMES
    FUNCTION = "run_async"
    CATEGORY = "BytePlus/MediaKit/Advanced"
    DESCRIPTION = ("Any AI MediaKit tool with a raw request body. For the common tasks, "
                   "use the dedicated BytePlus MediaKit nodes instead.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Remote API call: never reuse a cached result.
        return time.time()

    async def run_async(self, **kwargs):
        # Async so other BytePlus nodes in the workflow run at the same time.
        import asyncio
        return await asyncio.to_thread(self.run, **kwargs)

    def run(self, tool, params_json, resume_task_id="", api_key="", timeout_seconds=3600,
            poll_interval=10, download_results=True, delete_tos_after_run=True, **sockets):
        if resume_task_id and resume_task_id.strip():
            task = resume_task(resume_task_id, api_key, timeout_seconds, poll_interval)
            return _build_outputs(task, download_results)
        tool_id = _tool_from_label(tool)
        text = (params_json or "").strip() or "{}"
        try:
            body = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"params_json is not valid JSON: {exc}")
        if not isinstance(body, dict):
            raise ValueError("params_json must be a JSON object ({...}).")
        task = run_tool(tool_id, body, sockets, api_key, timeout_seconds, poll_interval,
                        delete_tos_after_run)
        return _build_outputs(task, download_results)


# The web extension reads tool templates and docs from here, so the catalog
# lives in exactly one place (mediakit_tools.json).
def _tools_payload():
    tools = {}
    for tool_id, spec in _catalog()["tools"].items():
        tools[_tool_label(tool_id, spec)] = {
            "id": tool_id,
            "title": spec["title"],
            "summary": spec["summary"],
            "doc": spec["doc"],
            "sync": spec["sync"],
            "template": spec["template"],
            "params": [{k: p[k] for k in ("name", "type", "required", "description")}
                       for p in spec["params"]],
        }
    return {"tools": tools}


try:
    from server import PromptServer
    from aiohttp import web as _aiohttp_web

    @PromptServer.instance.routes.get("/byteplus_mediakit/tools")
    async def _serve_mediakit_tools(request):
        return _aiohttp_web.json_response(_tools_payload())
except Exception as exc:
    print(f"{_LOG} Tool templates route unavailable ({exc}); params_json will not "
          "auto-fill when the tool changes.")


NODE_CLASS_MAPPINGS = {
    "BytePlusMediaKit": BytePlusMediaKit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BytePlusMediaKit": "BytePlus MediaKit All Tools (Advanced)",
}
