# ComfyUI – BytePlus Nodes (Seedance, Seedream 5.5, Seed Audio, AI MediaKit)

> **Disclaimer — Proof-of-Concept Demonstration.** This script is provided as a
> proof-of-concept demonstration for evaluation and reference purposes only. It is
> not a finished product and does not form part of any commercial offering or
> service. It is provided "as-is", without warranty, support, or guarantee of
> fitness for a particular purpose. Users should review the source code in full
> before deployment and are responsible for its configuration, operation, and
> outcomes — including API usage, data storage, associated costs, data handling,
> security, credentials, and billing — and for compliance with the applicable
> BytePlus terms of service and relevant laws. To the fullest extent permitted by
> law, the authors and contributors disclaim any liability arising from its use.


ComfyUI nodes for BytePlus, built like ComfyUI's partner nodes: one node per
task, a few options, and the result straight out as a video, image or audio.

- **Seedance** (video): Text to Video, Image to Video, Reference to Video
- **Seedream 5.5** (image): generation and editing with native transparent
  backgrounds, plus **Seedream 5.5 Layers** (split an image into a base + up
  to 99 layers, with text OCR)
- **Seedream 5.0** (image): Pro and Lite
- **Seed Audio 1.0**: speech, dialogue, music and sound effects
- **AI MediaKit**: video enhance, frame interpolation, subtitle removal, face
  blur, background removal, reframe to vertical, trim / join / add audio,
  frame extraction, voice separation, image upscale, highlight reels, video Q&A,
  plus an advanced node with all 74 MediaKit tools

## Quick start
1. Put this folder in `ComfyUI/custom_nodes/`.
2. Install the dependencies with ComfyUI's own Python:
   `pip install -r requirements.txt`
3. Restart ComfyUI.
4. Open **Settings (gear icon) > BytePlus > API keys** and paste your keys
   (see "API keys" below). They are stored in your own ComfyUI, never in this
   folder.
5. Double-click the canvas and search **BytePlus**, or browse the
   **BytePlus** menu.

Independent BytePlus nodes in one workflow run at the same time.

## Seedance All-in-One (Advanced): reference
The original Seedance node is kept as **BytePlus Seedance All-in-One
(Advanced)** for batches (one video per prompt line). The notes below describe
it; the simple Seedance nodes share the same limits and uploads.

### Models
`dreamina-seedance-2-5-260628` is the default. Per-model limits are enforced by
the node *before* a task is queued, so an unsupported combination fails
immediately with a clear message instead of erroring asynchronously minutes
later.

| | Seedance 2.5 | Seedance 2.0 | 2.0 fast / 2.0 mini |
|---|---|---|---|
| Resolution | 480p, 720p | 480p, 720p, 1080p, 4k | 480p, 720p |
| Duration | 4–30 s or `-1` | 4–15 s or `-1` | 4–15 s or `-1` |
| Reference images | 30 | 9 | 9 |
| Reference videos | 10 (2–30 s each, ≤30 s total) | 3 (2–15 s each, ≤15 s total) | 3 |
| Reference audio | 10 (2–30 s each, ≤30 s total) | 3 (2–15 s each, ≤15 s total) | 3 |
| Audio-only reference | yes | no (needs an image or video) | no |
| `output_format` | mp4, mov | mp4 | mp4 |

`duration = -1` lets the model pick the length. All models run at 24 fps.

The `resolution`, `duration` and `output_format` widgets **follow the selected
model**: pick Seedance 2.5 and the resolution dropdown offers only 480p/720p and
`mov` becomes selectable; switch to Seedance 2.0 and 1080p/4k reappear while
`output_format` locks to `mp4`. A value that the new model cannot use is
corrected on the spot (4k → 720p, duration 30 → 15).

This is done by `web/seedance_model_caps.js`, which reads the limits from the
node over `/byteplus_seedance/model_caps` — the table lives only in `MODEL_CAPS`
in `nodes.py`. A second extension, `web/seedance_dynamic_inputs.js`, grows the
reference sockets on demand (see "Reference media notes"). Both are convenience
layers: the node re-validates every request server-side, so the limits still
hold if the extensions do not load.

> If the dropdowns do not adapt, or every reference socket is shown at once,
> after a restart, ComfyUI did not pick up the `web/` folder. Add this line to
> `__init__.py` and restart:
> ```python
> WEB_DIRECTORY = "./web"
> ```

### Seedance 2.5 `ratio` constraint
2.5 classifies each request into a task type and accepts **only
`ratio = adaptive`** for first-frame / first+last-frame, video-editing and
video-extension tasks (video editing additionally requires `duration = -1`).

- **Frame mode** is detectable locally, so the node forces `adaptive` and logs it.
- **Editing / extension** are classified server-side from prompt intent, so the
  node only prints a warning when your prompt looks like one and `ratio` is not
  `adaptive`. Set `ratio` to `adaptive` for those prompts.

Because of this, `ratio` now defaults to `adaptive` (it was `16:9`).

## Install
1. Put this folder in `ComfyUI/custom_nodes/`.
2. Install deps (use ComfyUI's terminal so it hits the right Python):
   `pip install -r requirements.txt`
3. Restart ComfyUI.

## Set your keys
Use **Settings > BytePlus > API keys** (recommended). Alternatively, a `.env`
file still works:
1. In this node folder, rename `.env.example` to `.env`.
2. Fill in:
   - `BYTEPLUS_API_KEY=ark-xxxx`  (required, for generation)
   - `ARK_ACCESS_KEY`, `ARK_SECRET_KEY`, `TOS_BUCKET`, `TOS_REGION`,
     `TOS_ENDPOINT`  (needed for the reference VIDEO/AUDIO inputs and for
     "Upload to Asset Library"). `ARK_ACCESS_KEY`/`ARK_SECRET_KEY` are your
     BytePlus account AK/SK, shared by TOS and the Asset Library. The legacy
     `TOS_ACCESS_KEY`/`TOS_SECRET_KEY` names still work as a fallback.
3. Leave the node's `api_key` field BLANK - it reads .env automatically.

## Use
1. Add the node: double-click canvas -> search **BytePlus Seedance**.
2. Enter prompt(s): **one prompt per line** - each line becomes a separate video.
3. `batch_count`: how many videos per prompt line (variations). Total videos =
   lines x batch_count (max 10).
4. Optionally add reference inputs (see below). They are shared across all tasks.
5. Connect the **video** output -> a **Save Video** node. Run.

All tasks are submitted up front and generated in parallel on BytePlus, then
polled together. Outputs are lists, so a single Save Video node downstream saves
every result. If some tasks fail, the successful ones are still returned.

Outputs (lists): video, video_path, task_id. Seedance 2.5 with
`output_format = mov` saves `.mov` files; everything else saves `.mp4`.

## Reference order (@-references)
Order is preserved within each type, so your prompt can address assets
positionally:
- images: socket order, then batch order within a socket -> @Image1, @Image2, ...
- videos: socket order -> @Video1, @Video2, ...
- audio:  socket order -> @Audio1, @Audio2, ...
The node prints the exact mapping to the console on each run.

## Reference media notes
- **The reference sockets grow as you use them.** The node starts with one
  socket per kind (`ref_image_1`, `ref_video_1`, `ref_audio_1`); connect one and
  the next appears. Only connected sockets plus a single spare per kind are ever
  shown, so the node stays short instead of displaying all 50 at once. Nothing
  is lost - the slots all still exist, they just appear on demand, and an
  unconnected slot is skipped exactly as before.
- **There is a socket per reference the API accepts** - 30 images, 10 videos and
  10 audio clips, matching the Seedance 2.5 maxima, 50 in all.
- Reference images are sent inline (base64, full resolution) - no TOS needed.
  **Each image socket also accepts an IMAGE batch**: every frame of a connected
  batch becomes its own `@ImageN` reference, in batch order. Use one socket per
  image when they differ in size - a ComfyUI batch is a single tensor, so
  batching forces every frame to share one width/height, whereas separate
  sockets each keep their original dimensions. The create-task request retries
  transient network errors, so a brief
  blip no longer aborts the run. If you see a write/connection timeout, a large
  inline image over a slow uplink is the likely cause - turn on "Upload to Asset
  Library" (which sends images via TOS as short URLs) or check your connection.
- Reference VIDEOS and AUDIO are uploaded to your TOS bucket and passed as
  presigned URLs, so they require TOS_* keys in .env. Without them, video/audio
  inputs are refused (images and frames still work).
- Video/audio come from connected sockets. Audio is serialized with torchaudio
  (the library ComfyUI's own audio nodes use) - no custom encoder. If your
  ComfyUI can load audio, torchaudio is already present.
- Per-model reference limits are in the Models table above. Formats: videos
  mp4/mov (<=200 MB each), audio wav/mp3 (<=15 MB each). The node checks the
  counts; it does not check clip durations, so keep each reference clip within
  the model's per-clip and total-duration budget.
- Reference mode requires Seedance 2.5 or the 2.0 series. Seedance 2.5 is the
  only model that accepts **audio-only** reference input.

## Upload to Asset Library
The node has a single **Upload to Asset Library** checkbox. When it is on,
**every connected image and video asset** (`first_frame`, `last_frame`,
`ref_image_*`, `ref_video_*`) is routed through your private (virtual) asset
library. **Audio is never uploaded** — it always stays a raw TOS URL.

Seedance 2.5 and 2.0 screen reference media for deepfake / copyright risk. Media that
lives in your private asset library is pre-cleared, so the node references those
inputs as `asset://<id>` instead of raw URLs.

With the switch on, each image/video input is:
1. written to your TOS bucket (the Assets API only accepts a fetchable URL — so
   even reference *images*, which are normally sent inline as base64, need TOS
   for this),
2. added to the library with `CreateAsset`,
3. polled with `GetAsset` until `Status=Active` (uploads are queued and have no
   guaranteed SLA — the wait defaults to 600 s; raise `ARK_ASSET_TIMEOUT` in
   `.env` if you hit the limit),
4. passed to the model as `asset://<id>`, and
5. **deleted from the virtual asset library once generation completes** — always,
   on success or failure.

Slot order and the `@Image1 / @Video1 / @Audio1` numbering are unchanged — the
switch only swaps the URLs, not the positions.

Requirements:
- **Advanced Creation Rights** must be active on the account, and the first
  asset group needs the authorization letter signed in the ModelArk console.
- **AK/SK account keys** (`ARK_ACCESS_KEY` / `ARK_SECRET_KEY` in `.env`), not the
  `ark-` API key. These are the same keys TOS uses; the legacy
  `TOS_ACCESS_KEY` / `TOS_SECRET_KEY` names still work as a fallback.
- TOS must be configured (as for video/audio reference input).

### "asset ... is not found" at generation time
The asset library is **isolated per project and per account**, and the node uses
two different credentials: AK/SK to upload assets, and `BYTEPLUS_API_KEY` to
generate. If generation reports `The specified asset ... is not found` even
though the upload succeeded, either:
- the asset is still propagating — the node already retries for up to ~2 min
  after `Active`, so this usually resolves itself; or
- your `BYTEPLUS_API_KEY` and your AK/SK belong to **different accounts or
  projects**. Use keys from the same account, and set `ARK_ASSET_PROJECT` to the
  project your API key runs in (assets default to the `default` project).

Optional `.env` overrides: `ARK_ASSET_TIMEOUT` (seconds to wait for `Active`,
default `600`), `ARK_ASSET_GROUP_ID` (upload into an existing group),
`ARK_ASSET_GROUP_NAME` (default `comfyui-seedance`, created on first use),
`ARK_ASSET_PROJECT` (default `default` — assets are isolated per project),
`ARK_OPENAPI_HOST`, `ARK_OPENAPI_REGION` (default `ap-southeast-1`).
Seedance 2.5 and the 2.0 series reject reference media containing real human
faces unless it comes from an approved source - check the BytePlus docs before
using footage of people.

## API keys: Settings > BytePlus
Open ComfyUI Settings (gear icon) > **BytePlus > API keys** and paste:
ModelArk (Seedance, Seedream), Seed Audio, AI MediaKit, and the Access key /
Secret key used to upload connected videos and audio to TOS (plus the Session
token if your AK starts with AKTP, i.e. temporary keys). Keys are saved to
ComfyUI's `user/byteplus.env`, outside this node folder, so sharing the node
never shares your keys: everyone who installs it enters their own. They work
on the next run, no restart needed. A `.env` file (see `.env.example`) or
environment variables still work too.

## Simple nodes (partner style)
| Menu | Nodes | Key in `.env` |
|---|---|---|
| BytePlus > Seedance | Text to Video, Image to Video (first + optional last frame), Reference to Video (images / videos / audio that grow as you connect them; call them @Image1, @Video1, @Audio1 in the prompt) | `BYTEPLUS_API_KEY` |
| BytePlus > Seedream | Seedream 5.5 (1K-2K, 10 refs, `background` auto / transparent / opaque, outputs image + mask), Seedream 5.5 Layers (base image + up to 99 layers placed on the base canvas, layer masks, layer info JSON and OCR text), Seedream 5.0 Pro (1K-2K, 10 refs), Seedream 5.0 Lite (2K-4K, 14 refs, image sets) | `BYTEPLUS_API_KEY` |
| BytePlus > Seed Audio | Seed Audio 1.0: speech, dialogue, music, SFX; voice from the prompt, a preset, up to 3 cloned clips or an image | `BYTEPLUS_SEED_AUDIO_API_KEY` |

Picking a model only shows the resolutions and durations it supports. The
original Seedance node is still there as **Seedance All-in-One (Advanced)**
(one video per prompt line, batches), so existing workflows keep working.

## AI MediaKit
Simple nodes, one per task (like ComfyUI's partner nodes): connect the media,
pick a few options, get the result back. Less common options are under
"Show advanced". Find them under **BytePlus > MediaKit** or search by name.

| Menu | Nodes |
|---|---|
| Video | Video Enhance, Video Enhance (Fast), Frame Interpolation, Fix Stutter, Remove Subtitles / Text, Blur Faces (Video), Remove Background (Video), Reframe to Vertical |
| Edit | Trim Video, Join Videos, Add Audio to Video, Extract Frames |
| Audio | Separate Voice / Background |
| Image | Image Upscale / Enhance, Remove Background (Image) |
| AI | Highlight Reel, Ask About Video |
| Advanced | MediaKit All Tools: any of the 74 tools with a raw request body (`params_json`, auto-filled per tool), plus `resume_task_id` to collect a task that timed out |

Setup: add `BYTEPLUS_MEDIAKIT_API_KEY=...` to `.env` (AI MediaKit console >
Settings > API key; the ModelArk `ark-` key does not work here). Connected
media is uploaded to TOS, so the TOS keys above are needed too. Every result
file is also saved to `output/mediakit/<task_id>/` (BytePlus links expire in
24 h). The tool catalog is `mediakit_tools.json`; regenerate it with the
scripts in `tools/`.
