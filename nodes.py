"""
ComfyUI custom nodes for the BytePlus ModelArk video generation API (Seedance).

DISCLAIMER - PROOF-OF-CONCEPT DEMONSTRATION
This script is provided as a proof-of-concept demonstration for evaluation and
reference purposes only. It is not a finished product and does not form part of
any commercial offering or service. It is provided "as-is", without warranty,
support, or guarantee of fitness for a particular purpose. Users should review
this source code in full before deployment and are responsible for its
configuration, operation, and outcomes - including API usage, data storage,
associated costs, data handling, security, credentials, and billing - and for
compliance with the applicable BytePlus terms of service and relevant laws. To
the fullest extent permitted by law, the authors and contributors disclaim any
liability arising from its use. Use constitutes acceptance of these terms.

Async flow:
  1. POST .../contents/generations/tasks   -> returns {id}
  2. GET  .../contents/generations/tasks/{id} (poll) -> status + content.video_url
  3. Download the MP4 (URL expires ~24h).

Docs:
  Create: https://docs.byteplus.com/en/docs/ModelArk/1520757
  Query : https://docs.byteplus.com/en/docs/ModelArk/1521309
  List  : https://docs.byteplus.com/en/docs/ModelArk/1521675
  Assets: https://docs.byteplus.com/en/docs/ModelArk/2315856
  2.5   : https://docs.byteplus.com/en/docs/ModelArk/2607688  (must-read: task
          classification and the ratio/duration constraints it imposes)
"""

import os
import io
import sys
import json
import time
import base64
import hashlib
import hmac
import datetime

import numpy as np
from PIL import Image
import requests

# --- ComfyUI integration (degrade gracefully if run outside Comfy) ---
try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except Exception:
    OUTPUT_DIR = os.path.join(os.getcwd(), "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import comfy.utils
    _HAS_PBAR = True
except Exception:
    _HAS_PBAR = False

try:
    import comfy.model_management as _mm
    _HAS_MM = True
except Exception:
    _HAS_MM = False


def _check_interrupt():
    """Raise if the user pressed Cancel/Abort in ComfyUI. No-op if unavailable."""
    if _HAS_MM:
        # Raises comfy's InterruptProcessingException when a stop was requested.
        _mm.throw_exception_if_processing_interrupted()

# Native ComfyUI VIDEO type (wraps a file on disk so it can feed the built-in
# "Save Video" node without re-encoding). Available on recent ComfyUI builds.
try:
    from comfy_api.input_impl import VideoFromFile
    _HAS_VIDEO = True
except Exception:
    try:
        from comfy_api.input_impl.video_types import VideoFromFile
        _HAS_VIDEO = True
    except Exception:
        VideoFromFile = None
        _HAS_VIDEO = False


DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"

# Confirmed Seedance model IDs on the global ModelArk route. 2.5 is the default
# (first entry). Note the 2.0 mini model uses a different date suffix (260615)
# than the rest of the 2.0 series.
KNOWN_MODELS = [
    "dreamina-seedance-2-5-260628",
    "dreamina-seedance-2-0-260128",
    "dreamina-seedance-2-0-fast-260128",
    "dreamina-seedance-2-0-mini-260615",
]

RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"]
# The union across models; the per-model subset is enforced by MODEL_CAPS below.
# 1080p/4k are Seedance 2.0 (standard) only - 2.5, 2.0 Fast and 2.0 Mini top out
# at 720p.
RESOLUTIONS = ["480p", "720p", "1080p", "4k"]
# Container of the generated video. Seedance 2.5 only; mov is H.264 + yuv444p +
# PCM, meant for grading/keying/compositing round-trips.
OUTPUT_FORMATS = ["mp4", "mov"]

# Per-model limits, so an unsupported combination is caught before a task is
# queued instead of failing asynchronously minutes later.
#   resolutions/duration = None -> unknown model, skip local validation
#   duration              = (min, max) in seconds; -1 (auto) is always allowed
#   audio_only            = reference audio may be sent without an image/video
_CAPS_DEFAULT = {
    "resolutions": None,
    "duration": None,
    "max_images": 30, "max_videos": 10, "max_audio": 10,
    "audio_only": True,
    "output_format": False,
    "frames_force_adaptive": False,
}

_CAPS_2_5 = {
    "resolutions": ("480p", "720p"),
    "duration": (4, 30),
    "max_images": 30, "max_videos": 10, "max_audio": 10,
    "audio_only": True,
    "output_format": True,
    # 2.5 classifies each request into a task type and rejects any ratio other
    # than 'adaptive' for first-frame / first+last-frame, video editing and
    # video extension tasks.
    "frames_force_adaptive": True,
}

_CAPS_2_0 = {
    "resolutions": ("480p", "720p", "1080p", "4k"),
    "duration": (4, 15),
    "max_images": 9, "max_videos": 3, "max_audio": 3,
    "audio_only": False,
    "output_format": False,
    "frames_force_adaptive": False,
}

# Fast/Mini share 2.0's limits but cannot render above 720p.
_CAPS_2_0_SMALL = dict(_CAPS_2_0, resolutions=("480p", "720p"))

MODEL_CAPS = {
    "dreamina-seedance-2-5-260628": _CAPS_2_5,
    "dreamina-seedance-2-0-260128": _CAPS_2_0,
    "dreamina-seedance-2-0-fast-260128": _CAPS_2_0_SMALL,
    "dreamina-seedance-2-0-mini-260615": _CAPS_2_0_SMALL,
}


def run_in_parallel(node_cls):
    """Make a V3 node's blocking execute() async, so ComfyUI runs independent
    BytePlus nodes of one workflow at the same time (it awaits async nodes
    concurrently). The blocking work - HTTP calls and polling - moves to a
    worker thread; Cancel still works because _check_interrupt() reads a
    process-wide flag."""
    import asyncio
    import functools
    sync_execute = node_cls.execute.__func__

    @functools.wraps(sync_execute)
    async def execute(cls, *args, **kwargs):
        return await asyncio.to_thread(sync_execute, cls, *args, **kwargs)

    node_cls.execute = classmethod(execute)
    return node_cls


def _caps_for(model_id):
    """Limits for a model id. Falls back to a family match (so a newer date
    suffix still resolves), then to a permissive entry that skips validation."""
    if model_id in MODEL_CAPS:
        return MODEL_CAPS[model_id]
    if "seedance-2-5" in model_id:
        return _CAPS_2_5
    if "seedance-2-0-fast" in model_id or "seedance-2-0-mini" in model_id:
        return _CAPS_2_0_SMALL
    if "seedance-2-0" in model_id:
        return _CAPS_2_0
    return _CAPS_DEFAULT

TERMINAL_OK = {"succeeded"}
TERMINAL_BAD = {"failed", "expired", "cancelled", "canceled"}

# Where this node package lives on disk.
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Env vars checked for the API key (field blank -> look these up).
_ENV_KEYS = ("BYTEPLUS_API_KEY", "ARK_API_KEY", "MODELARK_API_KEY")


# ----------------------------- helpers --------------------------------- #

def _user_env_path():
    """Where Settings > BytePlus saves keys: ComfyUI's user folder, outside this
    node folder, so sharing the node never shares the keys."""
    try:
        import folder_paths
        return os.path.join(folder_paths.get_user_directory(), "byteplus.env")
    except Exception:
        return None


def _read_env_file():
    """Parse the .env files (KEY=value lines) this package reads, merged.

    Later files win: ComfyUI root .env, then this node folder's .env, then the
    keys saved from Settings > BytePlus (user/byteplus.env). Returns a dict.
    No external dependency - just simple line parsing.
    """
    values = {}
    candidates = []
    try:
        import folder_paths
        candidates.append(os.path.join(folder_paths.base_path, ".env"))
    except Exception:
        pass
    candidates.append(os.path.join(_NODE_DIR, ".env"))
    user_env = _user_env_path()
    if user_env:
        candidates.append(user_env)
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    values[k.strip()] = v.strip().strip('"').strip("'")
        except Exception as exc:
            print(f"[BytePlusSeedance] Could not read {path}: {exc}")
    return values


def _resolve_api_key(field_value=""):
    """Field value -> OS env var -> .env file (first non-empty wins)."""
    if field_value and field_value.strip():
        return field_value.strip()
    for env_name in _ENV_KEYS:
        val = os.environ.get(env_name, "").strip()
        if val:
            return val
    env_file = _read_env_file()
    for env_name in _ENV_KEYS:
        val = env_file.get(env_name, "").strip()
        if val:
            return val
    return ""


def _redact(text):
    """Strip anything resembling an API key from text before logging/raising."""
    if not text:
        return text
    import re
    text = re.sub(r"ark-[A-Za-z0-9\-]{6,}", "ark-***REDACTED***", text)
    text = re.sub(r"(?i)(authorization\"?\s*[:=]\s*\"?Bearer\s+)[^\"\s]+",
                  r"\1***REDACTED***", text)
    return text


def _single_image_data_url(image_hwc):
    """One [H,W,C] float-0..1 tensor -> full-resolution base64 PNG data URL.

    Reference images and frames are sent inline (no TOS, no downscaling); the
    original pixels are preserved.
    """
    arr = (image_hwc.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _save_image_png(image_hwc, path):
    """One [H,W,C] float-0..1 tensor -> full-resolution PNG on disk.

    Used only when the asset-library switch is on: CreateAsset needs a fetchable
    URL, so the image is written to a file and uploaded to TOS first. No
    downscaling or re-compression - the original pixels are preserved.
    """
    arr = (image_hwc.detach().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, format="PNG")


def _encode_audio(audio_obj, path):
    """Write a ComfyUI AUDIO dict to a file using the audio library ComfyUI
    itself uses (torchaudio), falling back to soundfile. No hand-rolled encoder.

    An AUDIO socket only carries a decoded waveform in memory, so it must be
    serialized to a file before it can be uploaded to TOS.
    """
    wf = audio_obj["waveform"]
    sr = int(audio_obj["sample_rate"])
    w = wf[0] if hasattr(wf, "dim") and wf.dim() == 3 else wf  # [C, T]
    try:
        import torchaudio
        torchaudio.save(path, w.cpu(), sr)
        return
    except Exception as exc_ta:
        try:
            import soundfile as sf
            sf.write(path, w.cpu().numpy().T, sr)  # [T, C]
            return
        except Exception as exc_sf:
            raise RuntimeError(
                "Encoding the connected AUDIO input needs torchaudio (which "
                "ComfyUI's own audio nodes use) or soundfile. Neither is "
                f"available. torchaudio: {exc_ta}; soundfile: {exc_sf}"
            )


def _build_content(prompt, *, frame_first_url=None, frame_last_url=None,
                   ref_image_urls=None, ref_video_urls=None, ref_audio_urls=None):
    """Build the ModelArk `content` array.

    Two mutually exclusive modes (per the API spec):
      - Reference mode: any of reference images / videos / audio are present.
        Images -> reference_image, videos -> reference_video, audio -> reference_audio.
      - Frame mode: first_frame / last_frame only.
    Image URLs are base64 data URLs; video/audio URLs are TOS presigned URLs.
    """
    content = [{"type": "text", "text": prompt}]
    ref_image_urls = ref_image_urls or []
    ref_video_urls = ref_video_urls or []
    ref_audio_urls = ref_audio_urls or []
    reference_mode = bool(ref_image_urls or ref_video_urls or ref_audio_urls)

    if reference_mode:
        for url in ref_image_urls:
            content.append({"type": "image_url", "image_url": {"url": url},
                            "role": "reference_image"})
        for url in ref_video_urls:
            content.append({"type": "video_url", "video_url": {"url": url},
                            "role": "reference_video"})
        for url in ref_audio_urls:
            content.append({"type": "audio_url", "audio_url": {"url": url},
                            "role": "reference_audio"})
        return content

    if frame_first_url is not None:
        content.append({"type": "image_url", "image_url": {"url": frame_first_url},
                        "role": "first_frame"})
    if frame_last_url is not None:
        content.append({"type": "image_url", "image_url": {"url": frame_last_url},
                        "role": "last_frame"})
    return content


# ------------------------- TOS (object storage) ------------------------ #
# Reference-video input requires a public URL. We upload the local clip to
# BytePlus TOS and hand the model a presigned GET URL. Credentials come from
# .env / env vars; if they are absent, video input is refused.

# AK/SK are the same BytePlus account keys used by both TOS and the Assets API,
# so ARK_ACCESS_KEY / ARK_SECRET_KEY are the unified names for both. The older
# TOS_ACCESS_KEY / TOS_SECRET_KEY names are still accepted as a fallback.
_TOS_ENV = {
    "ak": ("ARK_ACCESS_KEY", "TOS_ACCESS_KEY", "TOS_ACCESS_KEY_ID"),
    "sk": ("ARK_SECRET_KEY", "TOS_SECRET_KEY", "TOS_SECRET_ACCESS_KEY"),
    "bucket": ("TOS_BUCKET",),
    "region": ("TOS_REGION",),
    "endpoint": ("TOS_ENDPOINT",),
}

# Only for temporary (STS) credentials, whose access key starts with AKTP: the
# session token that comes with them. Permanent AK/SK pairs have none.
_SESSION_TOKEN_ENV = ("ARK_SESSION_TOKEN", "TOS_SECURITY_TOKEN", "BYTEPLUS_SESSION_TOKEN")


def _env_lookup(names):
    """Check OS env first, then the .env file, for the first matching name."""
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    env_file = _read_env_file()
    for n in names:
        v = env_file.get(n, "").strip()
        if v:
            return v
    return ""


def _resolve_tos_config():
    """Return (config_dict, missing_list). config has ak/sk/bucket/region/endpoint."""
    cfg = {k: _env_lookup(names) for k, names in _TOS_ENV.items()}
    # Defaults, so the AK/SK pair is all a user has to enter.
    if not cfg["bucket"]:
        cfg["bucket"] = "comfy-tos-bucket"
    if not cfg["region"]:
        cfg["region"] = "ap-southeast-1"
    if not cfg["endpoint"]:
        cfg["endpoint"] = f"https://tos-{cfg['region']}.bytepluses.com"
    missing = [k for k, v in cfg.items() if not v]
    return cfg, missing


def _validate_bucket_name(name):
    """TOS/S3 bucket naming: 3-63 chars, lowercase letters/digits/hyphens,
    must start and end with a letter or digit. Raises ValueError if invalid."""
    import re
    if not re.fullmatch(r"[a-z0-9][a-z0-9\-]{1,61}[a-z0-9]", name or ""):
        raise ValueError(
            f"Invalid TOS bucket name '{name}'. Bucket names must be 3-63 "
            "characters, lowercase letters/numbers/hyphens only, and start and "
            "end with a letter or number (no uppercase, no underscores). "
            "Suggested: 'comfy-tos-bucket'."
        )


_TOS_PREFIX = "comfyui-seedance/"
_LIFECYCLE_DONE = set()  # buckets we've already applied the backstop rule to


def _apply_backstop_lifecycle(client, bucket):
    """Apply a 1-day expiry rule on our prefix as a fallback sweep.

    The node deletes its reference files after each run; this only catches
    objects orphaned by a hard crash that skipped that cleanup. Applied at most
    once per bucket per session; failures are non-fatal (logged, not raised).
    """
    if bucket in _LIFECYCLE_DONE:
        return
    try:
        import tos
        import tos.models2 as m
        rule = m.BucketLifeCycleRule(
            id="comfyui-seedance-backstop",
            prefix=_TOS_PREFIX,
            status=tos.enum.StatusType.Status_Enable,
            expiration=m.BucketLifeCycleExpiration(days=1),
        )
        client.put_bucket_lifecycle(bucket, [rule])
        _LIFECYCLE_DONE.add(bucket)
        print(f"[BytePlusSeedance] TOS backstop lifecycle set (orphaned files "
              f"under '{_TOS_PREFIX}' expire after 1 day).")
    except Exception as exc:
        # Non-fatal: per-run deletion is the primary cleanup.
        print(f"[BytePlusSeedance] Backstop lifecycle not set "
              f"(per-run deletion still applies): {exc}")


def _ensure_bucket(client, bucket, region):
    """Create the bucket (private) if needed, and set the backstop lifecycle."""
    import tos
    _validate_bucket_name(bucket)
    try:
        client.head_bucket(bucket)
        _apply_backstop_lifecycle(client, bucket)
        return  # exists and accessible
    except tos.exceptions.TosServerError as exc:
        status = getattr(exc, "status_code", None)
        # 404 -> does not exist; 403 -> exists but not yours / no access
        if status == 403:
            raise RuntimeError(
                f"TOS bucket '{bucket}' exists but your keys cannot access it "
                "(or the name is taken by another account). Pick a different "
                "TOS_BUCKET name."
            )
        if status not in (404, None):
            raise
    print(f"[BytePlusSeedance] TOS bucket '{bucket}' not found - creating it...")
    client.create_bucket(bucket)
    print(f"[BytePlusSeedance] Created TOS bucket '{bucket}'.")
    _apply_backstop_lifecycle(client, bucket)


def _tos_client():
    """Build a TOS client from .env/env config, or raise a clear error."""
    cfg, missing = _resolve_tos_config()
    if missing:
        raise ValueError(
            "Connected videos / audio are uploaded to your TOS bucket, which needs "
            "your Access key / Secret key: add them in Settings (gear icon) > "
            f"BytePlus > API keys. Missing: {', '.join(missing)}."
        )
    try:
        import tos
    except Exception:
        raise RuntimeError("The 'tos' package is not installed. Run: "
                           "pip install tos  (see requirements.txt).")
    client = tos.TosClientV2(cfg["ak"], cfg["sk"], cfg["endpoint"], cfg["region"],
                             security_token=_env_lookup(_SESSION_TOKEN_ENV) or None)
    return client, cfg["bucket"]


def _tos_upload(local_path, expires=86400):
    """Upload a file to TOS; return (signed_get_url, bucket, key).

    The key is returned so the caller can delete the object after the run -
    these reference files are not reused, so they don't need to linger.
    """
    import tos
    client, bucket = _tos_client()
    try:
        _ensure_bucket(client, bucket, None)
        ext = os.path.splitext(local_path)[1].lower() or ".mp4"
        key = f"{_TOS_PREFIX}ref-{int(time.time()*1000)}-{os.getpid()}{ext}"
        client.put_object_from_file(bucket, key, local_path)
    except tos.exceptions.TosServerError as exc:
        # The SDK error is a long dict; say what to fix instead.
        msg = getattr(exc, "message", "") or ""
        if "Access Key" in msg or "Signature" in msg or getattr(exc, "status_code", 0) == 403:
            raise RuntimeError(
                "TOS rejected your Access key / Secret key "
                f"({msg or 'access denied'}). Check them in Settings > BytePlus > "
                "API keys: they are the IAM Access key ID (usually starts with AKLT) "
                "and its Secret access key, not an API key.") from None
        raise RuntimeError(f"Uploading to TOS failed [{getattr(exc, 'status_code', '?')}]: "
                           f"{msg or exc}") from None
    out = client.pre_signed_url(
        tos.HttpMethodType.Http_Method_Get, bucket, key, expires=int(expires)
    )
    return out.signed_url, bucket, key


def _tos_delete(objects):
    """Delete uploaded objects after a run. objects: list of (bucket, key).
    Non-fatal: logs problems instead of raising."""
    if not objects:
        return
    try:
        client, _ = _tos_client()
    except Exception as exc:
        print(f"[BytePlusSeedance] TOS cleanup skipped: {exc}")
        return
    for bucket, key in objects:
        try:
            client.delete_object(bucket, key)
        except Exception as exc:
            print(f"[BytePlusSeedance] Could not delete TOS object {key}: {exc}")
    print(f"[BytePlusSeedance] Deleted {len(objects)} reference file(s) from TOS.")


# --------------------- private asset library (OpenAPI) ----------------- #
# Seedance 2.5 and 2.0 screen reference media for deepfake / copyright risk. Assets
# that live in your *private asset library* are pre-cleared, so passing them as
# `asset://<id>` instead of a raw URL avoids that interception. The library is
# managed through the BytePlus OpenAPI (Action=CreateAsset, ...), which uses
# AK/SK signature-v4 auth - NOT the ark- API key used for generation.
#
# Flow per marked input: file -> TOS (public URL) -> CreateAsset -> poll
# GetAsset until Status=Active -> use asset://<id> -> DeleteAsset after the run.
#
# Docs: https://docs.byteplus.com/en/docs/ModelArk/2315856

_ASSET_SERVICE = "ark"
_ASSET_API_VERSION = "2024-01-01"
_ASSET_DEFAULT_REGION = "ap-southeast-1"
_ASSET_GROUP_DEFAULT_NAME = "comfyui-seedance"

# AK/SK for the OpenAPI - the same BytePlus account keys used by TOS, so both
# read the unified ARK_ACCESS_KEY / ARK_SECRET_KEY names. The legacy TOS_* names
# remain as a fallback so older .env files keep working.
_ASSET_ENV = {
    "ak": ("ARK_ACCESS_KEY", "BYTEPLUS_ACCESS_KEY", "VOLC_ACCESSKEY",
           "TOS_ACCESS_KEY", "TOS_ACCESS_KEY_ID"),
    "sk": ("ARK_SECRET_KEY", "BYTEPLUS_SECRET_KEY", "VOLC_SECRETKEY",
           "TOS_SECRET_KEY", "TOS_SECRET_ACCESS_KEY"),
}

_ASSET_GROUP_CACHE = {}  # (project, name) -> group id, resolved once per session

# How long to wait for an uploaded asset to finish preprocessing (Status=Active)
# before giving up. CreateAsset is queued with no guaranteed SLA, so this is
# generous. Not a node widget - override with ARK_ASSET_TIMEOUT (seconds) in .env.
try:
    _ASSET_TIMEOUT_SECS = float(_env_lookup(("ARK_ASSET_TIMEOUT",)) or 600)
except (TypeError, ValueError):
    _ASSET_TIMEOUT_SECS = 600.0


def _asset_credentials():
    """Return (ak, sk) for the OpenAPI, or raise with setup instructions."""
    ak = _env_lookup(_ASSET_ENV["ak"])
    sk = _env_lookup(_ASSET_ENV["sk"])
    if not ak or not sk:
        raise ValueError(
            "'Upload to Asset Library' needs BytePlus account keys (AK/SK) for "
            "the Assets OpenAPI. Add ARK_ACCESS_KEY and ARK_SECRET_KEY to your "
            ".env file (these are the same keys TOS uses; the legacy "
            "TOS_ACCESS_KEY / TOS_SECRET_KEY names still work as a fallback)."
        )
    return ak, sk


def _sign_v4(method, host, region, query, body, ak, sk):
    """BytePlus/Volcengine signature v4 -> request headers.

    Same shape as AWS SigV4 but with a 'request' terminator, an X-Date header
    and a mandatory X-Content-Sha256 header.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    x_date = now.strftime("%Y%m%dT%H%M%SZ")
    short_date = x_date[:8]
    payload_hash = hashlib.sha256(body).hexdigest()
    content_type = "application/json"

    # Temporary (STS) credentials also send and sign their session token.
    session_token = _env_lookup(_SESSION_TOKEN_ENV)
    signed_headers = "content-type;host;x-content-sha256;x-date"
    canonical_headers = (
        f"content-type:{content_type}\n"
        f"host:{host}\n"
        f"x-content-sha256:{payload_hash}\n"
        f"x-date:{x_date}\n"
    )
    if session_token:
        signed_headers += ";x-security-token"
        canonical_headers += f"x-security-token:{session_token}\n"
    from urllib.parse import quote
    canonical_query = "&".join(
        f"{quote(k, safe='-_.~')}={quote(query[k], safe='-_.~')}"
        for k in sorted(query)
    )
    canonical_request = "\n".join([
        method, "/", canonical_query, canonical_headers, signed_headers, payload_hash,
    ])

    scope = f"{short_date}/{region}/{_ASSET_SERVICE}/request"
    string_to_sign = "\n".join([
        "HMAC-SHA256", x_date, scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    def _hmac(key, msg):
        return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()

    k = _hmac(sk.encode("utf-8"), short_date)
    k = _hmac(k, region)
    k = _hmac(k, _ASSET_SERVICE)
    k = _hmac(k, "request")
    signature = hmac.new(k, string_to_sign.encode("utf-8"),
                         hashlib.sha256).hexdigest()

    headers = {
        "Content-Type": content_type,
        "Host": host,
        "X-Date": x_date,
        "X-Content-Sha256": payload_hash,
        "Authorization": (f"HMAC-SHA256 Credential={ak}/{scope}, "
                          f"SignedHeaders={signed_headers}, Signature={signature}"),
    }
    if session_token:
        headers["X-Security-Token"] = session_token
    return headers


def _asset_hosts(region):
    """Candidate OpenAPI hosts, best guess first.

    BytePlus documents the shared 'open.' gateway, while the Python SDK's
    resolver builds a per-service, per-region name. Which one answers depends on
    the account's region, so the first host that responds is cached and reused.
    Set ARK_OPENAPI_HOST in .env to pin one explicitly.
    """
    pinned = _env_lookup(("ARK_OPENAPI_HOST",))
    if pinned:
        return [pinned]
    # First host matches how the byteplus-go/python SDK resolvers build the
    # endpoint for the non-global 'ark' service; the generic open.* gateway and
    # a regional open.* form follow as fallbacks.
    return [
        f"{_ASSET_SERVICE}.{region}.byteplusapi.com",
        "open.byteplusapi.com",
        f"open.{region}.byteplusapi.com",
    ]


_ASSET_HOST_CACHE = {}  # region -> the host that answered


def _asset_api(action, payload):
    """Call an Assets OpenAPI action; return the Result object."""
    ak, sk = _asset_credentials()
    region = _env_lookup(("ARK_OPENAPI_REGION",)) or _ASSET_DEFAULT_REGION
    query = {"Action": action, "Version": _ASSET_API_VERSION}
    body = json.dumps(payload).encode("utf-8")

    known = _ASSET_HOST_CACHE.get(region)
    hosts = [known] if known else _asset_hosts(region)
    last_exc = None
    for host in hosts:
        headers = _sign_v4("POST", host, region, query, body, ak, sk)
        try:
            r = requests.post(f"https://{host}/", params=query, data=body,
                              headers=headers, timeout=60)
        except requests.RequestException as exc:
            last_exc = RuntimeError(f"{action}: could not reach {host}: {exc}")
            continue
        try:
            data = r.json()
        except Exception:
            last_exc = RuntimeError(
                f"{action} failed [{r.status_code}] at {host}: {_redact(r.text)}")
            continue
        err = (data.get("ResponseMetadata") or {}).get("Error") or {}
        code = err.get("Code") or ""
        if code in ("InvalidAction", "UnknownAction", "InvalidVersion",
                    "InvalidRequest") or r.status_code == 404:
            # Wrong gateway for this account - try the next candidate.
            last_exc = RuntimeError(f"{action} rejected by {host}: "
                                    f"{_redact(json.dumps(err))}")
            continue
        _ASSET_HOST_CACHE[region] = host
        if err or r.status_code >= 400:
            raise RuntimeError(f"{action} failed [{r.status_code}]: "
                               f"{_redact(json.dumps(err or data))}")
        return data.get("Result") or data
    raise last_exc or RuntimeError(f"{action} failed: no OpenAPI host responded.")


def _asset_project():
    return _env_lookup(("ARK_ASSET_PROJECT",)) or "default"


def _ensure_asset_group():
    """Return the asset group id to upload into, creating it once if needed.

    Pin an existing group with ARK_ASSET_GROUP_ID, or name one with
    ARK_ASSET_GROUP_NAME; otherwise a 'comfyui-seedance' group is reused/created.
    """
    pinned = _env_lookup(("ARK_ASSET_GROUP_ID",))
    if pinned:
        return pinned
    project = _asset_project()
    name = _env_lookup(("ARK_ASSET_GROUP_NAME",)) or _ASSET_GROUP_DEFAULT_NAME
    cache_key = (project, name)
    if cache_key in _ASSET_GROUP_CACHE:
        return _ASSET_GROUP_CACHE[cache_key]

    listed = _asset_api("ListAssetGroups", {
        "Filter": {"Name": name, "GroupType": "AIGC"},
        "PageNumber": 1, "PageSize": 10,
    })
    for item in (listed.get("Items") or []):
        # ListAssetGroups matches Name fuzzily - take an exact hit only.
        if item.get("Name") == name:
            _ASSET_GROUP_CACHE[cache_key] = item["Id"]
            return item["Id"]

    created = _asset_api("CreateAssetGroup", {
        "Name": name,
        "Description": "Auto-created by the ComfyUI BytePlus Seedance node.",
        "ProjectName": project,
    })
    gid = created["Id"]
    _ASSET_GROUP_CACHE[cache_key] = gid
    print(f"[BytePlusSeedance] Created asset group '{name}' ({gid}). Note: the "
          "first group on an account requires signing the authorization letter "
          "in the ModelArk console.")
    return gid


def _create_asset(url, asset_type, name=""):
    """CreateAsset from a publicly reachable URL; return the new asset id."""
    res = _asset_api("CreateAsset", {
        "GroupId": _ensure_asset_group(),
        "URL": url,
        "AssetType": asset_type,     # Image | Video | Audio
        "Name": name,
        "ProjectName": _asset_project(),
    })
    return res["Id"]


def _wait_assets_active(asset_ids, timeout_seconds):
    """Poll GetAsset until every id is Active. Raises if any fails/times out.

    CreateAsset is asynchronous (queued preprocessing, no SLA), and an asset is
    only usable for inference once Status == Active.
    """
    if not asset_ids:
        return
    pending = list(asset_ids)
    deadline = time.time() + timeout_seconds
    project = _asset_project()
    print(f"[BytePlusSeedance] Waiting for {len(pending)} asset(s) to become "
          "Active in the private asset library...")
    while pending:
        _check_interrupt()
        still = []
        for aid in pending:
            info = _asset_api("GetAsset", {"Id": aid, "ProjectName": project})
            status = info.get("Status")
            if status == "Active":
                continue
            if status == "Failed":
                raise RuntimeError(
                    f"Asset {aid} failed preprocessing in the asset library "
                    "(it may have been rejected by the content review). It "
                    "cannot be used for generation."
                )
            still.append(aid)
        pending = still
        if not pending:
            break
        if time.time() > deadline:
            raise RuntimeError(
                f"{len(pending)} asset(s) were still processing after "
                f"{timeout_seconds}s. Uploads are queued with no guaranteed "
                "SLA - raise ARK_ASSET_TIMEOUT in your .env and try again."
            )
        slept = 0.0
        while slept < 3.0:
            _check_interrupt()
            time.sleep(1.0)
            slept += 1.0
    print("[BytePlusSeedance] All assets are Active.")


def _delete_assets(asset_ids):
    """Remove this run's assets from the private library. Non-fatal on error."""
    if not asset_ids:
        return
    removed = 0
    for aid in asset_ids:
        try:
            _asset_api("DeleteAsset", {"Id": aid, "ProjectName": _asset_project()})
            removed += 1
        except Exception as exc:
            print(f"[BytePlusSeedance] Could not delete asset {aid}: {exc}")
    print(f"[BytePlusSeedance] Removed {removed}/{len(asset_ids)} asset(s) from "
          "the private asset library.")


# ----------------------- create / poll / download ---------------------- #

class _AssetNotReadyError(RuntimeError):
    """The inference endpoint could not yet resolve an asset:// reference.

    Raised for the retryable 'asset ... is not found' 400 so the caller can wait
    out the propagation lag between CreateAsset->Active and the asset becoming
    visible to generation.
    """


def _is_asset_not_found(text):
    if not text:
        return False
    low = text.lower()
    return "asset" in low and "is not found" in low


# (connect, read) seconds for the create-task POST. IMPORTANT: urllib3 keeps the
# CONNECT timeout on the socket while it writes the request body and only swaps
# to the read timeout before reading the response (see
# urllib3/connectionpool.py _make_request). So the connect slot - not the read
# slot - is what bounds uploading an inline base64 reference image. It must be
# generous enough to push a large image over a slow uplink, or the write times
# out. Override with ARK_CREATE_TIMEOUT (seconds) in .env.
try:
    _CREATE_TIMEOUT_SECS = float(_env_lookup(("ARK_CREATE_TIMEOUT",)) or 300)
except (TypeError, ValueError):
    _CREATE_TIMEOUT_SECS = 300.0
_CREATE_TIMEOUT = (_CREATE_TIMEOUT_SECS, _CREATE_TIMEOUT_SECS)
_SUBMIT_NET_RETRIES = 3   # transient connection/timeout retries for create-task


def _interruptible_sleep(seconds):
    """time.sleep in ~1s slices so ComfyUI Cancel stays responsive."""
    slept = 0.0
    while slept < seconds:
        _check_interrupt()
        chunk = min(1.0, seconds - slept)
        time.sleep(chunk)
        slept += chunk


def _submit_task(base_url, headers, body):
    """POST a create-task request; return the task id.

    Retries transient network failures (connection reset / write-or-read
    timeout). A network error here means the request was not acknowledged, so
    re-sending it does not risk a duplicate task; a 4xx/5xx HTTP reply is a real
    server response and is never retried this way.
    """
    url = f"{base_url}/contents/generations/tasks"
    last_exc = None
    for attempt in range(1, _SUBMIT_NET_RETRIES + 1):
        try:
            r = requests.post(url, headers=headers, json=body,
                              timeout=_CREATE_TIMEOUT)
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt >= _SUBMIT_NET_RETRIES:
                raise RuntimeError(
                    f"Create task network error after {attempt} attempts: {exc}. "
                    "Check your connection to the ModelArk endpoint; if it "
                    "persists, a reference image may be too large to upload."
                )
            wait = 2 ** attempt
            print(f"[BytePlusSeedance] create-task network error ({exc}); "
                  f"retry {attempt}/{_SUBMIT_NET_RETRIES - 1} in {wait}s...")
            _interruptible_sleep(wait)
    if r.status_code >= 400:
        if r.status_code == 400 and _is_asset_not_found(r.text):
            raise _AssetNotReadyError(_redact(r.text))
        raise RuntimeError(f"Create task failed [{r.status_code}]: {_redact(r.text)}")
    task = r.json()
    tid = task.get("id") or task.get("task_id")
    if not tid:
        raise RuntimeError(f"No task id in create response: {task}")
    return tid


def _poll_tasks(base_url, headers, task_ids, poll_interval, timeout_seconds,
                on_progress=None, on_success=None):
    """Poll many tasks together until each is done or the deadline passes.

    All tasks were already submitted, so BytePlus generates them in parallel;
    this just checks each one each cycle. The moment a task succeeds, on_success
    (tid, video_url) is called so the caller can download it immediately - this
    secures finished work right away (the result URLs are temporary) instead of
    waiting for the whole batch.

    Returns (succeeded_ids:set, errors:dict). on_progress(done, total) fires
    whenever the finished count changes.
    """
    pending = set(task_ids)
    succeeded, errors = set(), {}
    deadline = time.time() + timeout_seconds
    last_done = 0
    while pending:
        _check_interrupt()  # honor ComfyUI Cancel/Abort promptly
        if time.time() > deadline:
            for tid in pending:
                errors[tid] = f"did not finish within {timeout_seconds}s"
            break
        # Sleep in ~1s slices so Cancel is responsive even with a long interval.
        slept = 0.0
        while slept < poll_interval:
            _check_interrupt()
            chunk = min(1.0, poll_interval - slept)
            time.sleep(chunk)
            slept += chunk
        for tid in list(pending):
            try:
                g = requests.get(f"{base_url}/contents/generations/tasks/{tid}",
                                 headers=headers, timeout=60)
                if g.status_code >= 400:
                    errors[tid] = f"query failed [{g.status_code}]: {_redact(g.text)}"
                    pending.discard(tid)
                    continue
                data = g.json()
                status = (data.get("status") or "").lower()
                if status in TERMINAL_OK:
                    c = data.get("content") or {}
                    url = c.get("video_url") or c.get("url")
                    pending.discard(tid)
                    if not url:
                        errors[tid] = "succeeded but no video_url"
                    else:
                        # Secure the result now - URL is temporary.
                        try:
                            if on_success:
                                on_success(tid, url)
                            succeeded.add(tid)
                        except Exception as exc:
                            errors[tid] = f"download failed: {exc}"
                elif status in TERMINAL_BAD:
                    errors[tid] = f"{status}: {data.get('error') or ''}"
                    pending.discard(tid)
            except Exception as exc:
                # transient error - keep polling this task next cycle
                print(f"[BytePlusSeedance] poll error for {tid}: {exc}")
        done = len(succeeded) + len(errors)
        if done != last_done:
            last_done = done
            if on_progress:
                on_progress(done, len(task_ids))
        print(f"[BytePlusSeedance] progress: {done}/{len(task_ids)} finished")
    return succeeded, errors


def _download_video(url, task_id, default_ext=".mp4"):
    """Download the result video into the ComfyUI output dir; return its path.

    Seedance 2.5 can return .mov, so the extension is taken from the result URL
    when it carries one and from the requested output_format otherwise.
    """
    from urllib.parse import urlparse
    path = urlparse(url).path.lower()
    ext = next((e for e in (".mov", ".mp4") if path.endswith(e)), default_ext)
    out_path = os.path.join(OUTPUT_DIR, f"seedance_{task_id}{ext}")
    dl = requests.get(url, timeout=300)
    dl.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(dl.content)
    return out_path


# --------------------------- main node --------------------------------- #

# Every socket that carries an asset, in the order the API sees it. One socket
# per reference the API will accept - 30 images + 10 videos + 10 audio, the
# Seedance 2.5 maxima. That is 50 sockets, but web/seedance_dynamic_inputs.js
# only shows the connected ones plus a spare, so the node stays short.
#
# An IMAGE socket ALSO contributes every frame of a connected batch (see
# _iter_images), so a batch is an alternative to wiring 30 sockets. Use separate
# sockets when the images differ in size: a ComfyUI batch is one tensor, so
# batching forces every frame to share one width/height.
_IMAGE_SLOTS = [f"ref_image_{i}" for i in range(1, 31)]
_VIDEO_SLOTS = [f"ref_video_{i}" for i in range(1, 11)]
_AUDIO_SLOTS = [f"ref_audio_{i}" for i in range(1, 11)]
_FRAME_SLOTS = ["first_frame", "last_frame"]

# Words that make Seedance 2.5 classify a reference task as video editing or
# video extension - both of which reject any ratio but 'adaptive' (and editing
# also rejects any duration but -1). Detection happens server-side from prompt
# intent, so this is only used to warn before a task is queued.
_EDIT_HINTS = ("edit the video", "video edit", "remove ", "delete ", "replace ",
               "modify ", "erase ")
_EXTEND_HINTS = ("extend ", "continue the", "continue @", "keep going")


def _iter_images(tensor):
    """Yield each [H,W,C] frame of a connected IMAGE input, in batch order.

    A ComfyUI IMAGE socket carries a [B,H,W,C] batch, so one connection can hold
    several reference images; each becomes its own @ImageN reference.
    """
    if tensor is None:
        return
    if hasattr(tensor, "dim") and tensor.dim() == 4:
        for i in range(tensor.shape[0]):
            yield tensor[i]
    else:
        yield tensor


class BytePlusSeedanceVideo:
    """Text-to-video and image-to-video via the BytePlus ModelArk Seedance API."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        # --- Frame mode (mutually exclusive with reference mode) ---
        for slot in _FRAME_SLOTS:
            optional[slot] = ("IMAGE",)
        optional["timeout_seconds"] = ("INT", {"default": 600, "min": 30, "max": 3600})
        optional["delete_tos_after_run"] = ("BOOLEAN", {
            "default": True,
            "tooltip": "Delete the reference files from TOS once the run "
                       "finishes (they are not reused).",
        })
        # --- Reference image/video/audio sockets ---
        # Connect what you need; empty slots are skipped. Order follows the slot
        # numbers: ref_image_1 = @Image1, ref_video_1 = @Video1, ... An IMAGE
        # socket may carry a batch, and every frame in it becomes its own
        # @ImageN in batch order - that is how 2.5's 30-image limit is reached.
        for slots, sock in ((_IMAGE_SLOTS, "IMAGE"), (_VIDEO_SLOTS, "VIDEO"),
                            (_AUDIO_SLOTS, "AUDIO")):
            for slot in slots:
                optional[slot] = (sock,)

        # NOTE: ComfyUI stores widget values positionally (widgets_values), so a
        # new widget MUST be appended last. Inserting one in the middle shifts
        # every later widget's saved value by one in existing workflows - e.g.
        # output_format would show 'true' because it inherited generate_audio's
        # slot. Sockets do not occupy a widget slot, so declaring this after them
        # still puts it at the end of the widget list.
        optional["output_format"] = (OUTPUT_FORMATS, {
            "default": "mp4",
            "tooltip": "Seedance 2.5 only. mov is high-colour-precision "
                       "(H.264 + yuv444p + PCM) for grading/keying; some "
                       "players cannot open it.",
        })

        return {
            "optional": optional,
            "required": {
                # First widget on the node (above api_key). When on, every image
                # and video asset (frames and references, but NOT audio) is routed
                # through the private asset library and referenced as asset://<id>.
                "upload_to_asset_library": ("BOOLEAN", {
                    "default": False,
                    "label_on": "on",
                    "label_off": "off",
                    "tooltip": ("Upload all connected image and video assets to "
                                "your private (virtual) asset library and "
                                "reference them as asset://<id> instead of raw "
                                "URLs, then delete them from the library once "
                                "generation completes. Audio is never uploaded. "
                                "Needs AK/SK and TOS in .env."),
                }),
                "api_key": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": ("Leave blank to read the key from a .env file "
                                "(BYTEPLUS_API_KEY=...). Or paste a key here."),
                }),
                "model": (KNOWN_MODELS, {"default": KNOWN_MODELS[0]}),
                "prompt": ("STRING", {"default": "", "multiline": True,
                                      "tooltip": "One prompt per line. Each line is a "
                                                 "separate video, all generated in "
                                                 "parallel."}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 10,
                                        "tooltip": "Generate this many videos per prompt "
                                                   "line (variations)."}),
                "ratio": (RATIOS, {"default": "adaptive",
                    "tooltip": "Seedance 2.5 accepts only 'adaptive' for "
                               "first/last-frame, video-editing and video-"
                               "extension tasks - the node forces it for frame "
                               "mode and warns for the others."}),
                "resolution": (RESOLUTIONS, {"default": "720p",
                    "tooltip": "Seedance 2.5: 480p/720p only. 1080p/4k: Seedance "
                               "2.0 (standard) only. 2.0 Fast/Mini: 480p/720p."}),
                "duration": ("INT", {"default": 5, "min": -1, "max": 30, "step": 1,
                                     "tooltip": "Seconds. Seedance 2.5: 4-30. "
                                                "Seedance 2.0 series: 4-15. Use -1 "
                                                "to let the model choose (required "
                                                "for 2.5 video editing)."}),
                "generate_audio": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_path", "task_id")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "generate_async"
    CATEGORY = "BytePlus/Seedance/Advanced"
    DESCRIPTION = ("All Seedance modes in one node, with one video per prompt line and "
                   "batches. For everyday use, the Text / Image / Reference to Video "
                   "nodes are simpler.")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # This node calls a remote API; identical inputs should still produce a
        # fresh generation each run. Returning a unique value every time tells
        # ComfyUI never to reuse the cached output, so every queued Run fires
        # the API instead of being short-circuited by the cache.
        return time.time()

    async def generate_async(self, **kwargs):
        # Async so other BytePlus nodes in the workflow run at the same time;
        # the blocking work runs in a worker thread.
        import asyncio
        return await asyncio.to_thread(self.generate, **kwargs)

    def generate(self, api_key, model, prompt, batch_count, ratio, resolution, duration,
                 generate_audio, watermark=False, upload_to_asset_library=False,
                 output_format="mp4", first_frame=None, last_frame=None,
                 timeout_seconds=600, delete_tos_after_run=True, single_prompt=False,
                 **ref_slots):
        # single_prompt: the simple nodes in seedance_nodes.py send the whole
        # prompt as one task instead of one task per line.

        # One switch: route image+video assets through the private asset library
        # (audio always stays a raw TOS URL). Deletion afterward is unconditional.
        use_library = bool(upload_to_asset_library)

        # Internal constants (no longer exposed as inputs).
        custom_model = ""
        base_url = DEFAULT_BASE_URL
        poll_interval = 5.0

        resolved_key = _resolve_api_key(api_key)
        if not resolved_key:
            raise ValueError(
                "No API key. Add it in Settings (gear icon) > BytePlus > API keys, "
                "or paste one into the api_key field."
            )

        model_id = custom_model.strip() or model
        caps = _caps_for(model_id)

        # A workflow saved against an older widget layout can hand this combo a
        # value from a different widget (ComfyUI stores widget values by
        # position). Fall back rather than send garbage to the API.
        if output_format not in OUTPUT_FORMATS:
            print(f"[BytePlusSeedance] Ignoring unrecognised output_format "
                  f"'{output_format}' (expected one of {', '.join(OUTPUT_FORMATS)}) "
                  "- using mp4. Re-add the node if this workflow was saved "
                  "against an older version.")
            output_format = "mp4"

        # Reference sockets arrive as keyword args (there are ~30 of them), so
        # they are read by name in slot order rather than declared one by one.
        video_slots = [ref_slots.get(s) for s in _VIDEO_SLOTS]
        audio_slots = [ref_slots.get(s) for s in _AUDIO_SLOTS]
        # One IMAGE socket can carry a batch; every frame is its own reference.
        ref_images = []   # (label, [H,W,C] tensor) in @ImageN order
        for slot in _IMAGE_SLOTS:
            for j, hwc in enumerate(_iter_images(ref_slots.get(slot))):
                ref_images.append((slot if j == 0 else f"{slot}[{j}]", hwc))

        # ---- Determine mode and validate (frame vs reference are exclusive) ----
        frame_present = first_frame is not None or last_frame is not None
        ref_present = (bool(ref_images)
                       or any(v is not None for v in video_slots)
                       or any(a is not None for a in audio_slots))
        if frame_present and ref_present:
            raise ValueError(
                "first_frame/last_frame cannot be combined with reference "
                "images/videos/audio - these are mutually exclusive modes. "
                "Use one or the other."
            )

        # ---- Per-model limits, checked before anything is uploaded ----
        if caps["resolutions"] and resolution not in caps["resolutions"]:
            raise ValueError(
                f"{model_id} does not support {resolution}. Supported: "
                f"{', '.join(caps['resolutions'])}."
            )
        if caps["duration"] and duration != -1:
            lo, hi = caps["duration"]
            if not lo <= int(duration) <= hi:
                raise ValueError(
                    f"{model_id} supports a duration of {lo}-{hi}s, or -1 to let "
                    f"the model choose. Got {duration}."
                )
        if output_format != "mp4" and not caps["output_format"]:
            raise ValueError(
                f"output_format '{output_format}' is a Seedance 2.5 feature; "
                f"{model_id} only outputs mp4."
            )
        n_videos_in = sum(1 for v in video_slots if v is not None)
        n_audio_in = sum(1 for a in audio_slots if a is not None)
        for count, limit, what in ((len(ref_images), caps["max_images"], "images"),
                                   (n_videos_in, caps["max_videos"], "videos"),
                                   (n_audio_in, caps["max_audio"], "audio clips")):
            if count > limit:
                raise ValueError(
                    f"{model_id} accepts at most {limit} reference {what}; "
                    f"{count} are connected."
                )

        # 2.5 rejects any ratio but 'adaptive' for first/last-frame tasks - it
        # always matches the first frame's aspect ratio - so force it rather than
        # let the task fail asynchronously after it has queued.
        if caps["frames_force_adaptive"] and frame_present and ratio != "adaptive":
            print(f"[BytePlusSeedance] {model_id} only allows ratio='adaptive' "
                  f"for first/last-frame tasks; overriding '{ratio}'. The output "
                  "keeps the first frame's aspect ratio.")
            ratio = "adaptive"

        # The same constraint applies to video editing / extension, but 2.5
        # decides that from prompt intent server-side, so only warn.
        if caps["frames_force_adaptive"] and ref_present:
            low = (prompt or "").lower()
            editing = any(h in low for h in _EDIT_HINTS)
            extending = any(h in low for h in _EXTEND_HINTS)
            if (editing or extending) and ratio != "adaptive":
                print(f"[BytePlusSeedance] WARNING: this prompt reads like a "
                      f"video {'edit' if editing else 'extension'} task, which "
                      f"{model_id} only accepts with ratio='adaptive'. The task "
                      "will fail after queueing if it is classified that way.")
            if editing and duration != -1:
                print("[BytePlusSeedance] WARNING: video editing tasks only "
                      "accept duration=-1 (the output matches the input clip). "
                      f"duration={duration} will fail if the prompt is "
                      "classified as an edit.")

        ref_image_urls, ref_video_urls, ref_audio_urls = [], [], []
        frame_first_url, frame_last_url = None, None
        temp_files = []
        tos_objects = []   # (bucket, key) uploaded this run, for cleanup
        asset_ids = []     # asset library ids created this run, for cleanup

        # When the library switch is on, image/video jobs reserve their slot in
        # the content array up front; the asset://<id> (or TOS URL) is filled in
        # once TOS + CreateAsset have run. Audio is never sent to the library.
        jobs = []          # {"path", "slot", "type", "sink", "idx", "library"}
        frame_urls = [None, None]   # first_frame, last_frame

        def _reserve(sink, path, slot, asset_type, library):
            jobs.append({"path": path, "slot": slot, "type": asset_type,
                         "sink": sink, "idx": len(sink), "library": library})
            sink.append(None)

        def _cleanup_remote():
            """Drop everything this run staged remotely. Never reused, so it
            goes as soon as the run ends - success or failure. Assets are always
            removed from the virtual asset library."""
            if asset_ids:
                _delete_assets(asset_ids)
            if delete_tos_after_run and tos_objects:
                _tos_delete(tos_objects)

        try:
            import tempfile
            tmp_dir = tempfile.gettempdir()
            stamp = int(time.time() * 1000)

            if ref_present:
                # Images (slot order, batch order within a slot) -> inline base64
                # (full-res, no TOS). With the library switch on they go to disk
                # instead: CreateAsset needs a fetchable URL, so they are
                # uploaded to TOS + registered.
                for slot, hwc in ref_images:
                    if use_library:
                        safe = slot.replace("[", "_").replace("]", "")
                        tmp = os.path.join(tmp_dir, f"seedance_{stamp}_{safe}.png")
                        _save_image_png(hwc, tmp)
                        temp_files.append(tmp)
                        _reserve(ref_image_urls, tmp, slot, "Image", True)
                    else:
                        ref_image_urls.append(_single_image_data_url(hwc))

                # Videos (slot order) -> temp mp4; always uploaded to TOS, and
                # additionally into the asset library when the switch is on.
                n_videos = 0
                for slot, v in zip(_VIDEO_SLOTS, video_slots):
                    if v is None:
                        continue
                    tmp = os.path.join(tmp_dir, f"seedance_{stamp}_{slot}.mp4")
                    try:
                        v.save_to(tmp)
                    except Exception as exc:
                        raise RuntimeError(f"Could not read a connected video input: {exc}")
                    temp_files.append(tmp)
                    _reserve(ref_video_urls, tmp, slot, "Video", use_library)
                    n_videos += 1

                # Audio (slot order) -> encode to temp wav via torchaudio, upload
                # to TOS. Audio never enters the asset library (library=False).
                n_audio = 0
                for slot, a in zip(_AUDIO_SLOTS, audio_slots):
                    if a is None:
                        continue
                    tmp = os.path.join(tmp_dir, f"seedance_{stamp}_{slot}.wav")
                    try:
                        _encode_audio(a, tmp)
                    except Exception as exc:
                        raise RuntimeError(f"Could not read a connected audio input: {exc}")
                    temp_files.append(tmp)
                    _reserve(ref_audio_urls, tmp, slot, "Audio", False)
                    n_audio += 1

                # Seedance 2.5 accepts audio-only reference input; the 2.0 series
                # requires at least one reference image or video alongside it.
                if not (ref_image_urls or n_videos):
                    if not n_audio:
                        raise ValueError(
                            "Reference mode needs at least one reference image, "
                            "video or audio clip." if caps["audio_only"] else
                            "Reference mode needs at least one reference image "
                            "or video."
                        )
                    if not caps["audio_only"]:
                        raise ValueError(
                            f"{model_id} does not accept audio-only reference "
                            "input - include at least one reference image or "
                            "video (Seedance 2.5 does support audio alone)."
                        )

                manifest = (
                    [f"@Image{i+1}" for i in range(len(ref_image_urls))]
                    + [f"@Video{i+1}" for i in range(n_videos)]
                    + [f"@Audio{i+1}" for i in range(n_audio)]
                )
                if manifest:
                    print("[BytePlusSeedance] Reference order: " + ", ".join(manifest))
            else:
                # Frames (first/last) -> inline base64 (full-res, no TOS), or to
                # disk + TOS + asset library when the switch is on.
                for i, (slot, frame) in enumerate(zip(_FRAME_SLOTS,
                                                      (first_frame, last_frame))):
                    if frame is None:
                        continue
                    hwc = frame[0] if frame.dim() == 4 else frame
                    if use_library:
                        tmp = os.path.join(tmp_dir, f"seedance_{stamp}_{slot}.png")
                        _save_image_png(hwc, tmp)
                        temp_files.append(tmp)
                        jobs.append({"path": tmp, "slot": slot, "type": "Image",
                                     "sink": frame_urls, "idx": i, "library": True})
                    else:
                        frame_urls[i] = _single_image_data_url(hwc)

            # 1) Everything queued needs a fetchable URL -> TOS.
            if jobs:
                print(f"[BytePlusSeedance] Uploading {len(jobs)} reference file(s) "
                      "to TOS...")
                for job in jobs:
                    url, bucket, key = _tos_upload(job["path"])
                    tos_objects.append((bucket, key))
                    job["url"] = url
                    if not job["library"]:
                        job["sink"][job["idx"]] = url
                print("[BytePlusSeedance] Reference media uploaded.")

            # 2) Marked files additionally enter the private asset library, and
            #    are referenced as asset://<id> so Seedance treats them as
            #    pre-cleared instead of screening the raw URL.
            lib_jobs = [j for j in jobs if j["library"]]
            if lib_jobs:
                print(f"[BytePlusSeedance] Adding {len(lib_jobs)} file(s) to the "
                      "private asset library: "
                      + ", ".join(j["slot"] for j in lib_jobs))
                for job in lib_jobs:
                    aid = _create_asset(job["url"], job["type"], job["slot"])
                    asset_ids.append(aid)
                    job["sink"][job["idx"]] = f"asset://{aid}"
                    print(f"[BytePlusSeedance] {job['slot']} -> {aid}")
                _wait_assets_active(asset_ids, _ASSET_TIMEOUT_SECS)

            frame_first_url, frame_last_url = frame_urls
        except BaseException:
            # Nothing was submitted - don't leave staged files behind.
            _cleanup_remote()
            raise
        finally:
            # Local temp image/video/audio files are on TOS now; remove them.
            for p in temp_files:
                try:
                    os.remove(p)
                except Exception:
                    pass

        base_url = base_url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }

        # Build the task list: one prompt per non-empty line x batch_count.
        if single_prompt:
            prompt_lines = [(prompt or "").strip()]
        else:
            prompt_lines = [ln.strip() for ln in (prompt or "").splitlines() if ln.strip()]
        if not prompt_lines:
            prompt_lines = [""]  # allow empty prompt (e.g. pure image/video reference)
        total = len(prompt_lines) * int(batch_count)
        if total > 10:
            raise ValueError(f"Batch too large: {total} tasks (max 10). Reduce prompt "
                             f"lines or batch_count.")

        def _body_for(p):
            c = _build_content(
                p, frame_first_url=frame_first_url, frame_last_url=frame_last_url,
                ref_image_urls=ref_image_urls, ref_video_urls=ref_video_urls,
                ref_audio_urls=ref_audio_urls,
            )
            body = {"model": model_id, "content": c, "ratio": ratio,
                    "resolution": resolution, "duration": int(duration),
                    "generate_audio": bool(generate_audio),
                    "watermark": bool(watermark)}
            # Only Seedance 2.5 knows this key, and the request body is strictly
            # validated - sending it to an older model is an error.
            if caps["output_format"]:
                body["output_format"] = output_format
            return body

        # 1) Submit all tasks up front (server generates them in parallel).
        #    When asset:// references are involved, the inference endpoint can
        #    briefly 404 an asset that just became Active (CreateAsset is async
        #    with no SLA), so retry the batch over a short grace window.
        def _submit_all():
            out = []   # (task_id, prompt) in submission order
            for p in prompt_lines:
                for _ in range(int(batch_count)):
                    out.append((_submit_task(base_url, headers, _body_for(p)), p))
            return out

        try:
            if asset_ids:
                # A correctly set-up asset is usable the instant it is Active, so
                # this short window only rides out a brief lag. If it does not
                # resolve quickly the cause is structural (see the message
                # below), and retrying for minutes would just waste time and TOS.
                grace_deadline = time.time() + 30
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        submitted = _submit_all()
                        break
                    except _AssetNotReadyError as exc:
                        if time.time() >= grace_deadline:
                            raise RuntimeError(
                                "The asset reached Active in your library but the "
                                "generation endpoint reports it 'not found':\n"
                                f"  {exc}\n"
                                "The library uses two credentials that must line "
                                "up, so check, in order:\n"
                                "  1) Advanced Creation Rights are activated on "
                                "the account (the whole asset-library feature is "
                                "gated behind them).\n"
                                "  2) The first asset group's authorization letter "
                                "is signed in the ModelArk console (My assets > "
                                "Virtual Portrait).\n"
                                "  3) BYTEPLUS_API_KEY and the AK/SK that uploaded "
                                "the asset belong to the SAME account, and the "
                                "API key's project matches the asset's project - "
                                "set ARK_ASSET_PROJECT to that project (assets "
                                "default to 'default').\n"
                                "Until then, use images without a real human face "
                                "and leave 'Upload to Asset Library' off."
                            )
                        print(f"[BytePlusSeedance] Asset not visible to "
                              f"generation yet (attempt {attempt}); brief lag, "
                              "retrying...")
                        _interruptible_sleep(5.0)
            else:
                submitted = _submit_all()
            print(f"[BytePlusSeedance] Submitted {len(submitted)} task(s): "
                  f"{', '.join(t for t, _ in submitted)}")

            # 2) Poll all together; download each video the instant it succeeds so
            #    finished work is secured immediately (result URLs are temporary).
            task_ids = [t for t, _ in submitted]
            pbar = comfy.utils.ProgressBar(len(task_ids)) if _HAS_PBAR else None

            def _on_progress(done, total):
                if pbar:
                    pbar.update_absolute(done, total)

            saved_paths = {}  # task_id -> local mp4 path

            fallback_ext = (f".{output_format}" if caps["output_format"] else ".mp4")

            def _on_success(tid, url):
                out_path = _download_video(url, tid, fallback_ext)
                saved_paths[tid] = out_path
                print(f"[BytePlusSeedance] Saved -> {out_path}")

            succeeded, errors = _poll_tasks(base_url, headers, task_ids,
                                            poll_interval, timeout_seconds,
                                            on_progress=_on_progress,
                                            on_success=_on_success)

            # 3) Assemble outputs in submission order from what was saved.
            videos, paths, ids = [], [], []
            for tid, p in submitted:
                out_path = saved_paths.get(tid)
                if out_path:
                    videos.append(VideoFromFile(out_path) if _HAS_VIDEO else None)
                    paths.append(out_path)
                    ids.append(str(tid))

            if errors:
                msg = "; ".join(f"{t}: {m}" for t, m in errors.items())
                print(f"[BytePlusSeedance] {len(errors)} task(s) failed: {msg}")
            if not videos:
                raise RuntimeError("All tasks failed. " +
                                   "; ".join(f"{t}: {m}" for t, m in errors.items()))

            return (videos, paths, ids)
        finally:
            # Reference files aren't reused - remove them from TOS and from the
            # private asset library now that generation has finished.
            _cleanup_remote()


# --------------------- per-model widget limits (UI) -------------------- #
# INPUT_TYPES() is evaluated once when the class is registered, so the
# resolution / duration / output_format widgets cannot vary by model from Python
# alone. web/seedance_model_caps.js narrows them in the browser and reads the
# table below over this route, so the limits live in exactly one place. Purely
# cosmetic: generate() re-validates everything before a task is submitted.

def _model_caps_payload():
    return {
        "models": {
            m: {
                "resolutions": list(c["resolutions"] or []),
                "duration": list(c["duration"] or []),
                "output_format": bool(c["output_format"]),
            }
            for m, c in MODEL_CAPS.items()
        },
        "all_resolutions": RESOLUTIONS,
        "all_output_formats": OUTPUT_FORMATS,
        # How many sockets of each kind the node actually declares, so
        # seedance_dynamic_inputs.js grows into slots that exist rather than
        # keeping its own copy of these numbers.
        "slots": {
            "image": len(_IMAGE_SLOTS),
            "video": len(_VIDEO_SLOTS),
            "audio": len(_AUDIO_SLOTS),
        },
    }


try:
    from server import PromptServer
    from aiohttp import web as _aiohttp_web

    @PromptServer.instance.routes.get("/byteplus_seedance/model_caps")
    async def _serve_model_caps(request):
        return _aiohttp_web.json_response(_model_caps_payload())

    # Serve web/ without needing WEB_DIRECTORY in __init__.py. ComfyUI reads
    # EXTENSION_WEB_DIRS when it registers static routes, which happens after
    # custom nodes are imported. Harmless if the attribute is absent.
    _comfy_nodes = sys.modules.get("nodes")
    _web_dir = os.path.join(_NODE_DIR, "web")
    if (_comfy_nodes is not None
            and hasattr(_comfy_nodes, "EXTENSION_WEB_DIRS")
            and os.path.isdir(_web_dir)):
        _comfy_nodes.EXTENSION_WEB_DIRS.setdefault("comfyui-byteplus-seedance",
                                                   _web_dir)
except Exception as exc:
    # Running outside ComfyUI, or an incompatible server build - the node still
    # works, the widgets just show every model's options.
    print(f"[BytePlusSeedance] Per-model widget narrowing is unavailable ({exc}). "
          "The node still validates the model's limits when you run it.")


NODE_CLASS_MAPPINGS = {
    "BytePlusSeedanceVideo": BytePlusSeedanceVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BytePlusSeedanceVideo": "BytePlus Seedance All-in-One (Advanced)",
}
