"""
Simple Seedance nodes, shaped like ComfyUI's partner Seedance nodes: one node
per mode (text, image, references), and a model picker that only shows the
resolutions and durations that model supports.

They all run through BytePlusSeedanceVideo.generate() in nodes.py, so limits,
uploads, the asset library and polling behave exactly as on the All-in-One
node; these only choose which inputs to show.

DISCLAIMER - PROOF-OF-CONCEPT DEMONSTRATION: see nodes.py; the same terms apply.
"""

from comfy_api.latest import io as IO

from .nodes import (
    KNOWN_MODELS,
    MODEL_CAPS,
    BytePlusSeedanceVideo,
    _IMAGE_SLOTS,
    _VIDEO_SLOTS,
    _AUDIO_SLOTS,
    run_in_parallel,
)

CAT = "BytePlus/Seedance"

# Menu label -> model id, in KNOWN_MODELS order (2.5 first, the default).
MODEL_LABELS = {
    "dreamina-seedance-2-5-260628": "Seedance 2.5",
    "dreamina-seedance-2-0-260128": "Seedance 2.0",
    "dreamina-seedance-2-0-fast-260128": "Seedance 2.0 Fast",
    "dreamina-seedance-2-0-mini-260615": "Seedance 2.0 Mini",
}
MODELS = {MODEL_LABELS.get(m, m): m for m in KNOWN_MODELS}

RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"]

MODEL_TOOLTIP = ("Seedance 2.5: newest, 4-30 s, up to 30 reference images. "
                 "Seedance 2.0: up to 4K. Fast / Mini: quicker and cheaper, up to 720p.")


def _model_inputs(model_id, ratio=True, default_ratio="16:9"):
    """The widgets shown under one model choice, limited to what it supports."""
    caps = MODEL_CAPS[model_id]
    lo, hi = caps["duration"]
    resolutions = list(caps["resolutions"])
    inputs = [
        IO.String.Input("prompt", multiline=True, default="",
                        tooltip="Describe the video. Put spoken lines in double quotes."),
        IO.Combo.Input("resolution", options=resolutions,
                       default="720p" if "720p" in resolutions else resolutions[0]),
    ]
    if ratio:
        inputs.append(IO.Combo.Input("ratio", options=RATIOS, default=default_ratio,
                                     tooltip="'adaptive' follows the input media."))
    inputs += [
        IO.Int.Input("duration", default=5, min=lo, max=hi, step=1,
                     display_mode=IO.NumberDisplay.slider,
                     tooltip=f"Seconds ({lo}-{hi})."),
        IO.Boolean.Input("generate_audio", default=True,
                         tooltip="Generate sound and dialogue with the video."),
    ]
    if caps["output_format"]:
        inputs.append(IO.Combo.Input("output_format", options=["mp4", "mov"], default="mp4",
                                     advanced=True, optional=True,
                                     tooltip="mov is H.264 4:4:4 + PCM for grading and "
                                     "compositing; some players cannot open it."))
    return inputs


def _model_input(ratio_for=lambda m: True, default_ratio="16:9"):
    return IO.DynamicCombo.Input(
        "model",
        options=[IO.DynamicCombo.Option(label, _model_inputs(mid, ratio_for(mid), default_ratio))
                 for label, mid in MODELS.items()],
        tooltip=MODEL_TOOLTIP,
    )


def _common_advanced(asset_library=True):
    inputs = [IO.Boolean.Input("watermark", default=False, advanced=True, optional=True)]
    if asset_library:
        inputs.append(IO.Boolean.Input(
            "upload_to_asset_library", default=False, advanced=True, optional=True,
            tooltip="Send images and videos through your private asset library "
            "(asset://), needed for real human faces. Needs AK/SK + TOS in .env."))
    inputs.append(IO.Int.Input("timeout_seconds", default=900, min=60, max=7200,
                               advanced=True, optional=True))
    return inputs


def _generate(model, watermark=False, upload_to_asset_library=False, timeout_seconds=900,
              first_frame=None, last_frame=None, **slots):
    model_id = MODELS[model["model"]]
    prompt = (model.get("prompt") or "").strip()
    videos, _, _ = BytePlusSeedanceVideo().generate(
        api_key="", model=model_id, prompt=prompt, batch_count=1,
        ratio=model.get("ratio", "adaptive"), resolution=model["resolution"],
        duration=int(model["duration"]), generate_audio=bool(model["generate_audio"]),
        watermark=bool(watermark), upload_to_asset_library=bool(upload_to_asset_library),
        output_format=model.get("output_format", "mp4"), first_frame=first_frame,
        last_frame=last_frame, timeout_seconds=int(timeout_seconds), single_prompt=True,
        **slots)
    return IO.NodeOutput(videos[0])


class BytePlusSeedanceTextToVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BytePlusSeedanceTextToVideo",
            display_name="BytePlus Seedance Text to Video",
            category=CAT,
            description="Generates a video from a text prompt with Seedance 2.5 / 2.0.",
            search_aliases=["seedance", "byteplus", "t2v", "text to video", "generate video"],
            inputs=[_model_input()] + _common_advanced(asset_library=False),
            outputs=[IO.Video.Output()],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Remote generation: every run is a new video, never reuse the cache.
        import time
        return time.time()

    @classmethod
    def execute(cls, model, **kw):
        if not (model.get("prompt") or "").strip():
            raise ValueError("Write a prompt.")
        return _generate(model, **kw)


class BytePlusSeedanceImageToVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BytePlusSeedanceImageToVideo",
            display_name="BytePlus Seedance Image to Video",
            category=CAT,
            description="Animates an image: it becomes the first frame, and an optional "
            "last frame sets where the video ends.",
            search_aliases=["seedance", "byteplus", "i2v", "image to video", "animate",
                            "first frame", "last frame"],
            inputs=[
                IO.Image.Input("first_frame"),
                IO.Image.Input("last_frame", optional=True,
                               tooltip="Optional: the video ends on this image."),
                # 2.5 always follows the first frame's shape, so it shows no ratio.
                _model_input(ratio_for=lambda m: not MODEL_CAPS[m]["frames_force_adaptive"],
                             default_ratio="adaptive"),
            ] + _common_advanced(),
            outputs=[IO.Video.Output()],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        import time
        return time.time()

    @classmethod
    def execute(cls, first_frame, model, last_frame=None, **kw):
        return _generate(model, first_frame=first_frame, last_frame=last_frame, **kw)


def _autogrow(name, socket, slot_names):
    """A group of sockets that grows as you connect them (image_1, image_2...)."""
    return IO.Autogrow.Input(
        name,
        template=IO.Autogrow.TemplateNames(socket, names=slot_names, min=0),
    )


class BytePlusSeedanceReferenceToVideo(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BytePlusSeedanceReferenceToVideo",
            display_name="BytePlus Seedance Reference to Video",
            category=CAT,
            description="Generates, edits or extends a video from reference images, videos "
            "and audio. In the prompt, call them @Image1, @Video1, @Audio1... in the order "
            "they are connected.",
            search_aliases=["seedance", "byteplus", "reference", "r2v", "multimodal",
                            "edit video", "extend video", "character"],
            inputs=[
                _model_input(default_ratio="adaptive"),
                _autogrow("images", IO.Image.Input("image",
                          tooltip="Each frame of a batch is its own @ImageN."),
                          [f"image_{i}" for i in range(1, len(_IMAGE_SLOTS) + 1)]),
                _autogrow("videos", IO.Video.Input("video"),
                          [f"video_{i}" for i in range(1, len(_VIDEO_SLOTS) + 1)]),
                _autogrow("audios", IO.Audio.Input("audio"),
                          [f"audio_{i}" for i in range(1, len(_AUDIO_SLOTS) + 1)]),
            ] + _common_advanced(),
            outputs=[IO.Video.Output()],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        import time
        return time.time()

    @classmethod
    def execute(cls, model, images=None, videos=None, audios=None, **kw):
        slots = {}
        for group, names in ((images, _IMAGE_SLOTS), (videos, _VIDEO_SLOTS),
                             (audios, _AUDIO_SLOTS)):
            connected = [v for v in (group or {}).values() if v is not None]
            slots.update(zip(names, connected))
        if not slots:
            raise ValueError("Connect at least one reference image, video or audio.")
        return _generate(model, **slots, **kw)


NODES = [
    BytePlusSeedanceTextToVideo,
    BytePlusSeedanceImageToVideo,
    BytePlusSeedanceReferenceToVideo,
]

# Async, so independent BytePlus nodes in one workflow run at the same time.
NODES = [run_in_parallel(n) for n in NODES]
NODE_CLASS_MAPPINGS = {n.define_schema().node_id: n for n in NODES}
NODE_DISPLAY_NAME_MAPPINGS = {n.define_schema().node_id: n.define_schema().display_name
                              for n in NODES}
