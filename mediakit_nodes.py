"""
Simple BytePlus AI MediaKit nodes, one per task, shaped like ComfyUI's partner
API nodes: connect the media, pick a few options, get a VIDEO / IMAGE / AUDIO
back. Rarely needed options are marked advanced (hidden until "Show advanced").

All requests go through mediakit.run_tool(), which uploads the connected media
to TOS, calls the tool, waits for it and returns the task. Any tool not covered
here is available on "BytePlus MediaKit All Tools (Advanced)".

DISCLAIMER - PROOF-OF-CONCEPT DEMONSTRATION: see nodes.py; the same terms apply.
"""

import os

import numpy as np
from comfy_api.latest import io as IO

from .mediakit import (
    _empty_audio,
    _load_audio,
    _load_images,
    run_tool,
    save_results,
)
from .nodes import VideoFromFile, run_in_parallel

CAT = "BytePlus/MediaKit"

RESOLUTIONS_ENHANCE = ["source", "720p", "1080p", "2k", "4k", "8k"]
RESOLUTIONS_FAST = ["source", "480p", "720p", "1080p", "2k", "4k"]
FPS_PRESETS = ["source", "24", "25", "30", "50", "60", "120"]
ENHANCE_SCENES = ["aigc", "common", "ugc", "short_series", "old_film"]
HIGHLIGHT_PRESETS = ["custom", "football", "basketball", "automotive", "finance",
                     "product_launch", "ecommerce", "knowledge_share"]


# ----------------------------- helpers --------------------------------- #

def _advanced():
    """Inputs every node shares, hidden under "Show advanced"."""
    return [
        IO.Int.Input("timeout_seconds", default=3600, min=30, max=86400, optional=True,
                     advanced=True, tooltip="How long to wait for the task. It keeps running "
                     "on BytePlus after a timeout; collect it with the All Tools node's "
                     "resume_task_id."),
    ]


def _run(tool, body, sockets, kw):
    task = run_tool(tool, body, sockets, timeout_seconds=kw.get("timeout_seconds", 3600))
    return task, save_results(task)


def _files(files, kind, field=None):
    return [f["path"] for f in files
            if f["kind"] == kind and (field is None or f["field"] == field)]


def _video(files, tool):
    paths = _files(files, "video")
    if not paths:
        raise RuntimeError(f"AI MediaKit {tool} finished without a video in its result.")
    return VideoFromFile(paths[0])


def _images(files, tool):
    images = _load_images(_files(files, "image"))
    if images is None:
        raise RuntimeError(f"AI MediaKit {tool} finished without an image in its result.")
    return images


def _audio(files, field):
    paths = _files(files, "audio", field)
    return _load_audio(paths[0]) if paths else _empty_audio()


def _frames(image):
    """Each frame of an IMAGE batch as its own [1,H,W,C] tensor."""
    return [image[i:i + 1] for i in range(image.shape[0])]


def _video_node(node_id, name, category, description, inputs, aliases=()):
    return IO.Schema(
        node_id=node_id,
        display_name=name,
        category=f"{CAT}/{category}",
        description=description,
        search_aliases=["mediakit", "byteplus", *aliases],
        inputs=inputs + _advanced(),
        outputs=[IO.Video.Output()],
    )


# ------------------------------ video AI -------------------------------- #

class BytePlusVideoEnhance(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        style = lambda: IO.Combo.Input(
            "enhance_style", options=["hd", "natural"], default="hd",
            tooltip="'hd' is sharper; 'natural' is softer with fewer sharpening artifacts.")
        return _video_node(
            "BytePlusVideoEnhance", "BytePlus Video Enhance", "Video",
            "Upscales and restores a video (vCube): super-resolution up to 8K, removes "
            "compression artifacts and noise, improves colour and sharpness.",
            [
                IO.Video.Input("video", tooltip="Source up to 2K (2560x1440)."),
                IO.DynamicCombo.Input("tool_version", options=[
                    IO.DynamicCombo.Option("standard", [
                        IO.Combo.Input("scene", options=ENHANCE_SCENES, default="aigc",
                                       tooltip="'aigc' AI footage, 'common' general, 'ugc' phone "
                                       "clips, 'short_series' drama with faces, 'old_film' archive."),
                        style(),
                    ]),
                    IO.DynamicCombo.Option("professional", [style()]),
                ], tooltip="'professional' is cinema-grade, about 3x slower and 10x the price."),
                IO.Combo.Input("resolution", options=RESOLUTIONS_ENHANCE, default="1080p",
                               tooltip="Output short side. 'source' keeps the size."),
                IO.Combo.Input("fps", options=FPS_PRESETS, default="source",
                               tooltip="Above the source rate interpolates frames with AI."),
                IO.Combo.Input("bitrate_level", options=["low", "medium", "high"],
                               default="medium", advanced=True, optional=True),
            ],
            aliases=("upscale", "vcube", "super resolution", "restore"),
        )

    @classmethod
    def execute(cls, video, tool_version, resolution, fps, bitrate_level="medium", **kw):
        body = {"tool_version": tool_version["tool_version"],
                "enhance_style": tool_version.get("enhance_style", "hd"),
                "bitrate_level": bitrate_level}
        if tool_version["tool_version"] == "standard":
            body["scene"] = tool_version.get("scene", "common")
        if resolution != "source":
            body["resolution"] = resolution
        if fps != "source":
            body["fps"] = float(fps)
        _, files = _run("enhance-video", body, {"video": video}, kw)
        return IO.NodeOutput(_video(files, "enhance-video"))


class BytePlusVideoEnhanceFast(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _video_node(
            "BytePlusVideoEnhanceFast", "BytePlus Video Enhance (Fast)", "Video",
            "Quick, low-cost upscale and clean-up. Source short side 360-1440 px.",
            [
                IO.Video.Input("video"),
                IO.Combo.Input("resolution", options=RESOLUTIONS_FAST, default="1080p"),
                IO.Combo.Input("fps", options=FPS_PRESETS, default="source"),
                IO.Combo.Input("bitrate_level", options=["low", "medium", "high"],
                               default="medium", advanced=True, optional=True),
            ],
            aliases=("upscale", "fast"),
        )

    @classmethod
    def execute(cls, video, resolution, fps, bitrate_level="medium", **kw):
        body = {"bitrate_level": bitrate_level}
        if resolution != "source":
            body["resolution"] = resolution
        if fps != "source":
            body["fps"] = float(fps)
        _, files = _run("enhance-video-fast", body, {"video": video}, kw)
        return IO.NodeOutput(_video(files, "enhance-video-fast"))


class BytePlusFrameInterpolation(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _video_node(
            "BytePlusFrameInterpolation", "BytePlus Frame Interpolation", "Video",
            "Raises the frame rate with AI-generated in-between frames (smoother motion).",
            [
                IO.Video.Input("video"),
                IO.Int.Input("fps", default=60, min=15, max=120,
                             tooltip="Target frame rate. Best at no more than 4x the source."),
            ],
            aliases=("fps", "slow motion", "smooth"),
        )

    @classmethod
    def execute(cls, video, fps, **kw):
        _, files = _run("video-frame-interpolation", {"fps": fps}, {"video": video}, kw)
        return IO.NodeOutput(_video(files, "video-frame-interpolation"))


class BytePlusVideoSmoothness(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _video_node(
            "BytePlusVideoSmoothness", "BytePlus Fix Stutter", "Video",
            "Detects and repairs stutter and duplicated frames (videos up to 35 s, 4K).",
            [IO.Video.Input("video")],
            aliases=("stutter", "duplicate frames", "smoothness"),
        )

    @classmethod
    def execute(cls, video, **kw):
        _, files = _run("enhance-video-smoothness", {}, {"video": video}, kw)
        if not _files(files, "video"):
            # Nothing to repair: MediaKit returns no new video.
            return IO.NodeOutput(video)
        return IO.NodeOutput(_video(files, "enhance-video-smoothness"))


class BytePlusRemoveVideoText(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _video_node(
            "BytePlusRemoveVideoText", "BytePlus Remove Subtitles / Text", "Video",
            "Erases burned-in subtitles or all on-screen text and fills the area in.",
            [
                IO.Video.Input("video", tooltip="Up to 2K; output up to 1080p."),
                IO.Combo.Input("remove", options=["subtitles", "all text"], default="subtitles",
                               tooltip="'subtitles' only checks the lower half of the frame."),
                IO.Combo.Input("model_version", options=["v5", "v5.6"], default="v5",
                               advanced=True, optional=True,
                               tooltip="v5.6 reduces flicker and supports more languages."),
                IO.Combo.Input("priority", options=["Quality", "Size"], default="Quality",
                               advanced=True, optional=True),
            ],
            aliases=("erase", "subtitle", "watermark", "text removal"),
        )

    @classmethod
    def execute(cls, video, remove, model_version="v5", priority="Quality", **kw):
        body = {"mode": "Subtitle" if remove == "subtitles" else "Text",
                "model_version": model_version, "output_encode_mode": priority}
        _, files = _run("erase-video-subtitle-pro", body, {"video": video}, kw)
        return IO.NodeOutput(_video(files, "erase-video-subtitle-pro"))


class BytePlusFaceBlurVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _video_node(
            "BytePlusFaceBlurVideo", "BytePlus Blur Faces (Video)", "Video",
            "Detects every face and hides it with mosaic or blur (up to 10 min, 25-60 fps).",
            [
                IO.Video.Input("video"),
                IO.Combo.Input("style", options=["mosaic", "blur"], default="blur"),
                IO.Combo.Input("strength", options=["low", "medium", "high"], default="medium"),
                IO.Float.Input("face_confidence", default=0.35, min=0.1, max=1.0, step=0.05,
                               advanced=True, optional=True,
                               tooltip="Higher = fewer, surer detections."),
                IO.Float.Input("face_box_expand", default=0.2, min=0.01, max=1.0, step=0.05,
                               advanced=True, optional=True),
                IO.Boolean.Input("upright_faces_only", default=True, advanced=True, optional=True,
                                 tooltip="Off also finds rotated / upside-down faces (slower)."),
            ],
            aliases=("anonymize", "privacy", "mosaic"),
        )

    @classmethod
    def execute(cls, video, style, strength, face_confidence=0.35, face_box_expand=0.2,
                upright_faces_only=True, **kw):
        body = {"mask_mode": style, "mask_strength": strength,
                "face_confidence": face_confidence, "face_box_expand": face_box_expand,
                "upright_face_only": upright_faces_only}
        _, files = _run("face-blur-video", body, {"video": video}, kw)
        return IO.NodeOutput(_video(files, "face-blur-video"))


class BytePlusRemoveVideoBackground(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _video_node(
            "BytePlusRemoveVideoBackground", "BytePlus Remove Background (Video)", "Video",
            "Cuts out the person (or keys a green screen) and outputs a transparent or "
            "solid-colour background.",
            [
                IO.Video.Input("video"),
                IO.Combo.Input("subject", options=["person", "green screen"], default="person"),
                IO.DynamicCombo.Input("output", options=[
                    IO.DynamicCombo.Option("transparent (MOV)", []),
                    IO.DynamicCombo.Option("transparent (WEBM)", []),
                    IO.DynamicCombo.Option("solid colour (MP4)", [
                        IO.Combo.Input("background", options=["green", "black", "white"],
                                       default="green"),
                    ]),
                ]),
            ],
            aliases=("matting", "rotoscope", "alpha", "chroma key"),
        )

    @classmethod
    def execute(cls, video, subject, output, **kw):
        choice = output["output"]
        fmt = "MOV" if "MOV" in choice else "WEBM" if "WEBM" in choice else "MP4"
        body = {"format": fmt}
        if fmt == "MP4":
            body["background_color"] = output.get("background", "green")
        tool = "matte-portrait-video" if subject == "person" else "matte-greenscreen-video"
        _, files = _run(tool, body, {"video": video}, kw)
        return IO.NodeOutput(_video(files, tool))


class BytePlusReframeVertical(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _video_node(
            "BytePlusReframeVertical", "BytePlus Reframe to Vertical", "Video",
            "Turns a landscape video (16:9, 4:3, 21:9) into 9:16, following the subject.",
            [IO.Video.Input("video")],
            aliases=("9:16", "portrait", "tiktok", "reels", "shorts"),
        )

    @classmethod
    def execute(cls, video, **kw):
        _, files = _run("reframe-video-to-vertical", {}, {"video": video}, kw)
        return IO.NodeOutput(_video(files, "reframe-video-to-vertical"))


# ------------------------------- editing -------------------------------- #

class BytePlusTrimVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _video_node(
            "BytePlusTrimVideo", "BytePlus Trim Video", "Edit",
            "Keeps the part of the video between start and end.",
            [
                IO.Video.Input("video"),
                IO.Float.Input("start", default=0.0, min=0.0, max=86400.0, step=0.1,
                               tooltip="Seconds."),
                IO.Float.Input("end", default=0.0, min=0.0, max=86400.0, step=0.1,
                               tooltip="Seconds; 0 = until the end."),
            ],
            aliases=("cut", "clip"),
        )

    @classmethod
    def execute(cls, video, start, end, **kw):
        body = {"start_time": round(start, 2)}
        if end > 0:
            if end <= start:
                raise ValueError("end must be after start (or 0 for the end of the video).")
            body["end_time"] = round(end, 2)
        _, files = _run("trim-video", body, {"video": video}, kw)
        return IO.NodeOutput(_video(files, "trim-video"))


class BytePlusJoinVideos(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _video_node(
            "BytePlusJoinVideos", "BytePlus Join Videos", "Edit",
            "Joins clips one after another. The output takes the first clip's resolution.",
            [
                IO.Video.Input("video_1"),
                IO.Video.Input("video_2"),
                IO.Video.Input("video_3", optional=True),
                IO.Video.Input("video_4", optional=True),
            ],
            aliases=("concat", "concatenate", "merge", "stitch"),
        )

    @classmethod
    def execute(cls, video_1, video_2, video_3=None, video_4=None, **kw):
        clips = [v for v in (video_1, video_2, video_3, video_4) if v is not None]
        slots = ["video", "video_2", "video_3", "video_4"][:len(clips)]
        body = {"video_urls": ["{" + s + "}" for s in slots]}
        _, files = _run("concat-video", body, dict(zip(slots, clips)), kw)
        return IO.NodeOutput(_video(files, "concat-video"))


class BytePlusAddAudioToVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return _video_node(
            "BytePlusAddAudioToVideo", "BytePlus Add Audio to Video", "Edit",
            "Puts an audio track (music, voice-over) on a video.",
            [
                IO.Video.Input("video"),
                IO.Audio.Input("audio"),
                IO.Boolean.Input("keep_original_audio", default=False,
                                 tooltip="On mixes the new audio with the video's own sound."),
                IO.Combo.Input("length", options=["match video", "match audio", "longest"],
                               default="match video"),
                IO.Combo.Input("fit_by", options=["trim", "speed"], default="trim",
                               advanced=True, optional=True,
                               tooltip="How to match lengths: cut the longer one or change speed."),
            ],
            aliases=("mux", "music", "soundtrack", "voiceover"),
        )

    @classmethod
    def execute(cls, video, audio, keep_original_audio, length, fit_by="trim", **kw):
        body = {"is_audio_reserve": keep_original_audio,
                "is_video_audio_sync": length != "longest"}
        if length != "longest":
            body["sync_mode"] = "video" if length == "match video" else "audio"
            body["sync_method"] = fit_by
        _, files = _run("mux-audio-video", body, {"video": video, "audio": audio}, kw)
        return IO.NodeOutput(_video(files, "mux-audio-video"))


class BytePlusExtractFrames(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BytePlusExtractFrames",
            display_name="BytePlus Extract Frames",
            category=f"{CAT}/Edit",
            description="Grabs frames at a fixed interval or on every scene change.",
            search_aliases=["mediakit", "byteplus", "snapshot", "thumbnail", "keyframes"],
            inputs=[
                IO.Video.Input("video"),
                IO.DynamicCombo.Input("mode", options=[
                    IO.DynamicCombo.Option("every N seconds", [
                        IO.Float.Input("seconds", default=1.0, min=0.01, max=3600.0, step=0.1),
                    ]),
                    IO.DynamicCombo.Option("scene changes", []),
                    IO.DynamicCombo.Option("first and last", []),
                ]),
                IO.Int.Input("max_frames", default=24, min=1, max=1000),
                IO.Int.Input("long_side", default=0, min=0, max=4096, advanced=True,
                             optional=True, tooltip="Resize the long side; 0 keeps the size."),
            ] + _advanced(),
            outputs=[IO.Image.Output("images")],
        )

    @classmethod
    def execute(cls, video, mode, max_frames, long_side=0, **kw):
        choice = mode["mode"]
        if choice == "every N seconds":
            body = {"snapshot_type": "TimeInterval", "time_interval": mode.get("seconds", 1.0),
                    "snapshot_limit": max_frames}
        elif choice == "scene changes":
            body = {"snapshot_type": "SceneChange", "snapshot_limit": max_frames}
        else:
            body = {"snapshot_type": "SpecifiedFrames", "specified_frames": [0, -1]}
        if long_side:
            body["scale_long"] = long_side
        _, files = _run("extract-frames", body, {"video": video}, kw)
        return IO.NodeOutput(_images(files, "extract-frames"))


# -------------------------------- audio --------------------------------- #

class BytePlusSeparateVoice(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BytePlusSeparateVoice",
            display_name="BytePlus Separate Voice / Background",
            category=f"{CAT}/Audio",
            description="Splits the voice from the music and background sound of a video "
            "or audio clip.",
            search_aliases=["mediakit", "byteplus", "stems", "vocals", "karaoke", "isolate"],
            inputs=[
                IO.Video.Input("video", optional=True, tooltip="Connect a video or an audio."),
                IO.Audio.Input("audio", optional=True),
                IO.Combo.Input("content", options=["general", "music", "drama", "narration"],
                               default="general",
                               tooltip="'drama' and 'narration' also split music from sound "
                               "effects."),
            ] + _advanced(),
            outputs=[
                IO.Audio.Output("voice"),
                IO.Audio.Output("background", tooltip="Background, or music for drama/narration."),
                IO.Audio.Output("effects", tooltip="Sound effects (drama/narration only)."),
            ],
        )

    @classmethod
    def execute(cls, content, video=None, audio=None, **kw):
        if video is None and audio is None:
            raise ValueError("Connect a video or an audio.")
        scene = {"general": "Audio", "music": "Music", "drama": "Drama",
                 "narration": "Narrate"}[content]
        sockets = {"video": video} if video is not None else {"audio": audio}
        _, files = _run("separate-voice", {"scene": scene, "output_format": "wav"}, sockets, kw)
        background = _audio(files, "background_audio_url")
        if not _files(files, "audio", "background_audio_url"):
            background = _audio(files, "music_audio_url")
        return IO.NodeOutput(_audio(files, "voice_audio_url"), background,
                             _audio(files, "sfx_audio_url"))


# -------------------------------- image --------------------------------- #

class BytePlusImageEnhance(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BytePlusImageEnhance",
            display_name="BytePlus Image Upscale / Enhance",
            category=f"{CAT}/Image",
            description="Upscales and restores an image. Each image in a batch is "
            "processed separately.",
            search_aliases=["mediakit", "byteplus", "upscale", "super resolution", "restore"],
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input("quality", options=["standard", "professional", "max"],
                               default="standard",
                               tooltip="'professional' adds natural detail; 'max' is "
                               "generative restoration for very poor images."),
                IO.Float.Input("scale", default=2.0, min=1.0, max=30.0, step=0.5,
                               tooltip="Magnification. Standard allows up to 8x."),
            ] + _advanced(),
            outputs=[IO.Image.Output()],
        )

    @classmethod
    def execute(cls, image, quality, scale, **kw):
        if quality == "standard" and scale > 8:
            raise ValueError("The standard version upscales at most 8x.")
        out = []
        for frame in _frames(image):
            body = {"tool_version": quality, "multiple": round(scale, 2)}
            _, files = _run("enhance-image", body, {"image": frame}, kw)
            out.append(_images(files, "enhance-image"))
        return IO.NodeOutput(_stack(out))


class BytePlusRemoveImageBackground(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BytePlusRemoveImageBackground",
            display_name="BytePlus Remove Background (Image)",
            category=f"{CAT}/Image",
            description="Cuts the subject out of an image; returns the image and its mask.",
            search_aliases=["mediakit", "byteplus", "matting", "cutout", "rembg", "alpha"],
            inputs=[
                IO.Image.Input("image"),
                IO.Combo.Input("subject", options=["general", "human", "product"],
                               default="general"),
                IO.Boolean.Input("crop_to_subject", default=False, advanced=True, optional=True,
                                 tooltip="Human / product only: crop away the empty border."),
            ] + _advanced(),
            outputs=[IO.Image.Output(), IO.Mask.Output("mask")],
        )

    @classmethod
    def execute(cls, image, subject, crop_to_subject=False, **kw):
        import torch
        from PIL import Image
        images, masks = [], []
        for frame in _frames(image):
            body = {"scene": subject, "output_format": "png"}
            if crop_to_subject and subject != "general":
                body["need_crop_background"] = True
            _, files = _run("remove-image-background", body, {"image": frame}, kw)
            paths = _files(files, "image")
            if not paths:
                raise RuntimeError("Remove background finished without an image.")
            rgba = np.asarray(Image.open(paths[0]).convert("RGBA"), dtype=np.float32) / 255.0
            images.append(torch.from_numpy(rgba[..., :3]).unsqueeze(0))
            # ComfyUI masks mark what to remove: 1 - alpha, like Load Image.
            masks.append(torch.from_numpy(1.0 - rgba[..., 3]).unsqueeze(0))
        return IO.NodeOutput(_stack(images), _stack(masks))


def _stack(tensors):
    """Concatenate per-frame results; frames of another size are resized to
    the first, because a ComfyUI batch is a single tensor."""
    import torch
    import torch.nn.functional as F
    first = tensors[0]
    out = [first]
    for t in tensors[1:]:
        if t.shape[1:3] != first.shape[1:3]:
            if t.dim() == 4:
                t = F.interpolate(t.movedim(-1, 1), size=first.shape[1:3],
                                  mode="bilinear").movedim(1, -1)
            else:
                t = F.interpolate(t.unsqueeze(1), size=first.shape[1:3],
                                  mode="bilinear").squeeze(1)
        out.append(t)
    return torch.cat(out, dim=0)


# --------------------------------- AI ----------------------------------- #

class BytePlusVideoHighlights(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BytePlusVideoHighlights",
            display_name="BytePlus Highlight Reel",
            category=f"{CAT}/AI",
            description="Finds the best moments of a long video with an LLM and cuts them "
            "into a highlight reel.",
            search_aliases=["mediakit", "byteplus", "highlights", "recap", "summary video"],
            inputs=[
                IO.Video.Input("video", tooltip="Up to 3 hours."),
                IO.Combo.Input("preset", options=HIGHLIGHT_PRESETS, default="custom",
                               tooltip="A scene preset, or 'custom' to describe it in prompt."),
                IO.String.Input("prompt", multiline=True, default="",
                                tooltip="What counts as a highlight (required for 'custom')."),
                IO.Int.Input("duration", default=60, min=5, max=3600,
                             tooltip="Length of the reel in seconds."),
                IO.Combo.Input("language", options=["en-US", "zh-CN"], default="en-US",
                               advanced=True, optional=True,
                               tooltip="Language of the clip titles and summaries."),
            ] + _advanced(),
            outputs=[IO.Video.Output(), IO.String.Output("clips",
                     tooltip="Title, time range and reason for each selected clip.")],
        )

    @classmethod
    def execute(cls, video, preset, prompt, duration, language="en-US", **kw):
        body = {"target_duration": [duration], "output_language": language,
                "video_urls": ["{video}"]}
        if preset != "custom":
            body["preset"] = preset
        if prompt.strip():
            body["scoring_prompt"] = prompt.strip()
        if "preset" not in body and "scoring_prompt" not in body:
            raise ValueError("Pick a preset or describe the highlights in prompt.")
        task, files = _run("video-highlights-llm", body, {"video": video}, kw)
        lines = []
        for output in (task.get("result") or {}).get("outputs") or []:
            for seg in output.get("segments") or []:
                spans = ", ".join(f"{s.get('source_start_ms', 0) / 1000:.1f}-"
                                  f"{s.get('source_end_ms', 0) / 1000:.1f}s"
                                  for s in seg.get("spans") or [])
                lines.append(f"{seg.get('title', '')} [{spans}]: {seg.get('reason', '')}")
        return IO.NodeOutput(_video(files, "video-highlights-llm"), "\n".join(lines))


class BytePlusVideoUnderstanding(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BytePlusVideoUnderstanding",
            display_name="BytePlus Ask About Video",
            category=f"{CAT}/AI",
            description="Answers a question about a video (summary, events with timestamps, "
            "moderation, tagging...).",
            search_aliases=["mediakit", "byteplus", "describe", "caption", "analyze", "vlm"],
            inputs=[
                IO.Video.Input("video"),
                IO.String.Input("prompt", multiline=True,
                                default="Summarize the video and list the key events with "
                                "their time ranges."),
                IO.Combo.Input("level", options=["Economy", "Balanced", "Quality"],
                               default="Balanced",
                               tooltip="Economy is fastest and cheapest; Quality is most detailed."),
            ] + _advanced(),
            outputs=[IO.String.Output("text")],
        )

    @classmethod
    def execute(cls, video, prompt, level, **kw):
        if not prompt.strip():
            raise ValueError("prompt is empty.")
        body = {"video_urls": ["{video}"], "prompt": prompt.strip(), "level": level}
        task = run_tool("video-understand-router", body, {"video": video},
                        timeout_seconds=kw.get("timeout_seconds", 3600))
        contents = (task.get("result") or {}).get("contents") or []
        return IO.NodeOutput("\n\n".join(str(c) for c in contents))


NODES = [
    BytePlusVideoEnhance,
    BytePlusVideoEnhanceFast,
    BytePlusFrameInterpolation,
    BytePlusVideoSmoothness,
    BytePlusRemoveVideoText,
    BytePlusFaceBlurVideo,
    BytePlusRemoveVideoBackground,
    BytePlusReframeVertical,
    BytePlusTrimVideo,
    BytePlusJoinVideos,
    BytePlusAddAudioToVideo,
    BytePlusExtractFrames,
    BytePlusSeparateVoice,
    BytePlusImageEnhance,
    BytePlusRemoveImageBackground,
    BytePlusVideoHighlights,
    BytePlusVideoUnderstanding,
]

# Async, so independent BytePlus nodes in one workflow run at the same time.
NODES = [run_in_parallel(n) for n in NODES]
NODE_CLASS_MAPPINGS = {n.define_schema().node_id: n for n in NODES}
NODE_DISPLAY_NAME_MAPPINGS = {n.define_schema().node_id: n.define_schema().display_name
                              for n in NODES}
