"""
Settings > BytePlus: save the API keys from inside ComfyUI.

Keys are written to ComfyUI's user folder (user/byteplus.env), NOT to this node
folder, so sharing the node (zip, git) never shares anyone's keys: each person
enters their own in their own ComfyUI. nodes._read_env_file() reads that file
on every run, so a saved key works without restarting.

The routes are write-only for keys: a response never contains a key, only
whether it is set, where it comes from and its last four characters. Only the
ComfyUI page itself may call them.
"""

import json
import os
from urllib.parse import urlparse

from .nodes import _NODE_DIR, _read_env_file, _user_env_path

ROUTE = "/byteplus/credentials"
_MAX_BODY = 32 * 1024

# credential id -> env var names it saves (several for a pair such as AK/SK).
CREDENTIALS = {
    "modelark": ["BYTEPLUS_API_KEY"],
    "seed_audio": ["BYTEPLUS_SEED_AUDIO_API_KEY"],
    "mediakit": ["BYTEPLUS_MEDIAKIT_API_KEY"],
    "storage": ["ARK_ACCESS_KEY", "ARK_SECRET_KEY"],
    "session_token": ["ARK_SESSION_TOKEN"],
}

# Other names the nodes also accept, so status reflects an older .env too.
_ALIASES = {
    "BYTEPLUS_API_KEY": ["ARK_API_KEY", "MODELARK_API_KEY"],
    "BYTEPLUS_SEED_AUDIO_API_KEY": ["SEED_AUDIO_API_KEY", "BYTEPLUS_SEED_SPEECH_API_KEY"],
    "BYTEPLUS_MEDIAKIT_API_KEY": ["MEDIAKIT_API_KEY", "BYTEPLUS_VOD_MEDIAKIT_API_KEY"],
    "ARK_ACCESS_KEY": ["TOS_ACCESS_KEY", "TOS_ACCESS_KEY_ID"],
    "ARK_SECRET_KEY": ["TOS_SECRET_KEY", "TOS_SECRET_ACCESS_KEY"],
    "ARK_SESSION_TOKEN": ["TOS_SECURITY_TOKEN", "BYTEPLUS_SESSION_TOKEN"],
}


def _hint(value):
    return f"...{value[-4:]}" if len(value) > 8 else "set"


def _read_file(path):
    values = {}
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def _write_file(path, updates):
    """Set (value) or remove (None) keys in path, keeping every other line."""
    lines = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    done = set()
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in updates:
            if updates[key] is not None and key not in done:
                out.append(f"{key}={updates[key]}")
            done.add(key)
            continue
        out.append(line)
    for key, value in updates.items():
        if key not in done and value is not None:
            out.append(f"{key}={value}")
    if not lines:
        out.insert(0, "# BytePlus keys saved from ComfyUI Settings > BytePlus. Do not share.")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def status():
    user_path = _user_env_path()
    saved = _read_file(user_path)
    merged = _read_env_file()
    result = {}
    for cred, names in CREDENTIALS.items():
        entries = []
        for name in names:
            candidates = [name] + _ALIASES.get(name, [])
            env = next((os.environ[n] for n in candidates if os.environ.get(n, "").strip()), "")
            file_val = next((merged[n] for n in candidates if merged.get(n, "").strip()), "")
            value = env or file_val
            source = ("environment" if env else
                      "settings" if saved.get(name) else
                      "file" if file_val else None)
            entries.append({"name": name, "configured": bool(value),
                            "hint": _hint(value) if value else "", "source": source,
                            "in_settings": bool(saved.get(name))})
        result[cred] = {
            "configured": all(e["configured"] for e in entries),
            "entries": entries,
            "removable": any(e["in_settings"] for e in entries),
        }
    return {"credentials": result, "file": user_path,
            "node_env": os.path.join(_NODE_DIR, ".env")}


def save(cred, values=None, clear=False):
    if cred not in CREDENTIALS:
        raise ValueError(f"Unknown credential '{cred}'.")
    path = _user_env_path()
    if not path:
        raise RuntimeError("ComfyUI's user folder is not available.")
    names = CREDENTIALS[cred]
    if clear:
        _write_file(path, {n: None for n in names})
        return status()
    values = values or {}
    updates = {}
    for name in names:
        v = str(values.get(name, "")).strip()
        if not v:
            raise ValueError(f"{name} is empty.")
        if any(c in v for c in "\r\n\0") or len(v) > 8192:
            raise ValueError(f"{name} is not a valid key.")
        updates[name] = v
    _write_file(path, updates)
    return status()


def _same_origin(request):
    """Reject requests a page on another site makes to this local server.
    Browsers send Sec-Fetch-Site and pages cannot forge it; without it, the
    Origin must match the Host. No Origin at all = not a browser page."""
    site = request.headers.get("Sec-Fetch-Site")
    if site:
        return site.lower() in ("same-origin", "none")
    origin = request.headers.get("Origin")
    if not origin:
        return True
    hosts = {request.headers.get("Host", "").lower()}
    hosts |= {h.strip().lower() for h in request.headers.get("X-Forwarded-Host", "").split(",")}
    hosts.discard("")
    return urlparse(origin).netloc.lower() in hosts


try:
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get(ROUTE)
    async def _get_status(request):
        if not _same_origin(request):
            return web.json_response({"error": "Forbidden origin."}, status=403)
        return web.json_response(status())

    @PromptServer.instance.routes.post(ROUTE)
    async def _post_save(request):
        if not _same_origin(request):
            return web.json_response({"error": "Forbidden origin."}, status=403)
        raw = await request.content.read(_MAX_BODY + 1)
        if len(raw) > _MAX_BODY:
            return web.json_response({"error": "Request too large."}, status=413)
        try:
            body = json.loads(raw or b"{}")
            data = save(body.get("credential"), body.get("values"), bool(body.get("clear")))
        except (ValueError, RuntimeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response(data)
except Exception as exc:
    print(f"[BytePlus] Settings > BytePlus is unavailable ({exc}); use a .env file instead.")
