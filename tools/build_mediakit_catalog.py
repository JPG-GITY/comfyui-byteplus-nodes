import json, os, re, sys, glob
SRC, OUT = sys.argv[1], sys.argv[2]
DOC = "https://docs.byteplus.com/en/docs/byteplus-vod/"

def clean(s):
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = s.replace("\\-", "-").replace("\\_", "_").replace("\\*", "*").replace("**", "")
    return re.sub(r"\s+", " ", s).strip()

def cells(line):
    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    return parts

def body_table(md):
    m = re.search(r"\n#{2,3} Body(?: parameters)?\s*\n(.*?)(?=\n#{2,4} )", md, re.S)
    if not m:
        return []
    rows = []
    for line in m.group(1).splitlines():
        if not line.startswith("|") or "---" in line or "**Parameter**" in line or line.startswith("|Parameter"):
            continue
        c = cells(line)
        if len(c) < 5:
            continue
        name = c[0].strip("`").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.\[\]]+", name):
            continue
        rows.append({
            "name": name, "type": clean(c[1]), "required": clean(c[2]).lower().startswith("yes"),
            "example": clean(c[3]).strip("`"), "description": clean(" ".join(c[4:]))[:600],
        })
    return rows

def examples(md):
    out = []
    for block in re.findall(r"```[^\n]*\n(.*?)```", md, re.S):
        m = re.search(r"-d\s+'(.*?)'\s*$", block.strip(), re.S) or re.search(r"--data(?:-raw)?\s+'(.*?)'", block, re.S)
        if not m:
            continue
        try:
            out.append(json.loads(m.group(1)))
        except Exception:
            try:
                out.append(json.loads(re.sub(r"//.*", "", m.group(1))))
            except Exception:
                pass
    return out

def category(tool, sync):
    if sync: return "Image"
    if tool in ("call-maas-proxy",): return "AI / MaaS"
    if tool in ("video-highlights-llm", "video-understand-router", "drama-script", "evaluate-aigc-video-quality", "segment-scenes", "semantic-segment", "semantic-segment-pro", "assess-video-quality", "detect-voice-activity"):
        return "Analysis / AI"
    if any(k in tool for k in ("audio", "voice")) and "video" not in tool:
        return "Audio"
    if tool in ("enhance-video", "enhance-video-fast", "enhance-video-smoothness", "video-frame-interpolation", "martencode-video", "erase-video-subtitle-pro", "face-blur-video", "matte-portrait-video", "matte-greenscreen-video", "reframe-video-to-vertical"):
        return "Video AI"
    return "Video editing"

tools = {}
for path in sorted(glob.glob(os.path.join(SRC, "*.md"))):
    slug = os.path.basename(path)[:-3]
    md = open(path).read()
    m = re.search(r"`POST https://mediakit\.[^/]+/api/v1/(tools(?:-sync)?)/([a-z0-9-]+)`", md)
    if not m:
        continue
    sync = m.group(1) == "tools-sync"
    tool = m.group(2)
    title = md.splitlines()[0].lstrip("# ").strip()
    title = re.sub(r"^Create (an?|the) ", "", title)
    title = re.sub(r" task$", "", title)
    params = body_table(md)
    ex = examples(md)
    intro = ""
    after = md[m.end():]
    for para in after.split("\n\n"):
        p = clean(para)
        if p and not p.startswith("INFO") and not p.startswith("#"):
            intro = p[:400]; break
    tools[tool] = {
        "title": title[:1].upper() + title[1:], "path": f"/{m.group(1)}/{tool}", "sync": sync,
        "category": category(tool, sync), "doc": DOC + slug, "summary": intro,
        "params": params, "examples": ex,
    }
order = ["Video AI", "Video editing", "Audio", "Image", "Analysis / AI", "AI / MaaS"]
tools = dict(sorted(tools.items(), key=lambda kv: (order.index(kv[1]["category"]), kv[0])))
json.dump({"base_urls": {"ap-southeast-1": "https://mediakit.ap-southeast-1.bytepluses.com/api/v1"},
           "tools": tools}, open(OUT, "w"), indent=1, ensure_ascii=False)
for t, v in tools.items():
    print(f"{v['category']:14} {t:38} sync={int(v['sync'])} params={len(v['params']):2} ex={len(v['examples'])}")
