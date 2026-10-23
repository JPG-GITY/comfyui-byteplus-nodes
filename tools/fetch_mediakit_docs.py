import json, re, sys, urllib.request, concurrent.futures as cf, os
OUT = sys.argv[1]
slugs = """ai-mediakit-basic-concepts-and-prerequisites ai-mediakit-get-task-details ai-mediakit-event-notification ai-mediakit-error-codes ai-mediakit-call-maas-proxy ai-mediakit-overview ai-mediakit-pricing ai-mediakit-release-notes
generate-images-with-modelark-via-ai-mediakit generate-videos-with-modelark-via-ai-mediakit generate-audio-with-seed-speech-via-ai-mediakit transcribe-audio-with-seed-speech-via-ai-mediakit""".split()
html = open(sys.argv[2]).read()
slugs += sorted(set(re.findall(r'byteplus-vod/(ai-mediakit-create-[a-zA-Z0-9-]+)', html)))
def get(slug):
    if os.path.exists(os.path.join(OUT, slug + ".md")): return slug, None, None
    import time
    for attempt in range(5):
        try: return _get(slug)
        except Exception as e: err=e; time.sleep(3)
    print("FAIL", slug, err); return slug, None, None
def _get(slug):
    url = f"https://docs.byteplus.com/en/docs/byteplus-vod/{slug}"
    import subprocess; h = subprocess.run(["curl","-sS","-m","60",url],capture_output=True,text=True,check=True).stdout
    m = re.search(r'window\._ROUTER_DATA\s*=\s*(\{.*?\})\s*</script>', h, re.S)
    data = json.loads(m.group(1))
    for v in data["loaderData"].values():
        if isinstance(v, dict) and v.get("curDoc"):
            d = v["curDoc"]; return slug, d.get("Title",""), d.get("MDContent","")
    return slug, "", ""
with cf.ThreadPoolExecutor(3) as ex:
    for slug, title, md in ex.map(get, slugs):
        if md is None: continue
        open(os.path.join(OUT, slug + ".md"), "w").write(f"# {title}\n\n{md}")
        print(slug, len(md))
