"""
Simple Seed Audio 1.0 node, shaped like ComfyUI's partner Seed Audio node:
speech, dialogue, music and sound effects from one prompt, with an optional
preset voice, cloned voices (up to 3 clips) or a voice derived from an image.

  POST https://voice.ap-southeast-1.bytepluses.com/api/v3/tts/create
  -> audio (base64), up to 120 s per request

Seed Audio is a BytePlus Voice product with its OWN key (X-Api-Key), not the
ModelArk ark- key: console > Voice > activate "Seed-Audio 1.0" > API key.

DISCLAIMER - PROOF-OF-CONCEPT DEMONSTRATION: see nodes.py; the same terms apply.
"""

import base64
import io

import numpy as np
import requests
import torch
from PIL import Image
from comfy_api.latest import io as IO

from .nodes import _check_interrupt, _env_lookup, _redact, run_in_parallel

CAT = "BytePlus/Seed Audio"
SEED_AUDIO_URL = "https://voice.ap-southeast-1.bytepluses.com/api/v3/tts/create"
_KEY_ENV = ("BYTEPLUS_SEED_AUDIO_API_KEY", "SEED_AUDIO_API_KEY", "BYTEPLUS_SEED_SPEECH_API_KEY")

MODE_TEXT = "describe in prompt"
MODE_PRESET = "preset voice"
MODE_CLONE = "clone from audio"
MODE_IMAGE = "voice from image"

# (speaker id, label) for built-in voices; ids are account-scoped.
PRESET_VOICES = [
    ("en_female_dacey_uranus_bigtts", "Dacey (Female, English)"),
    ("en_female_stokie_uranus_bigtts", "Stokie (Female, English)"),
    ("en_male_tim_uranus_bigtts", "Tim (Male, English)"),
    ("zh_female_vv_uranus_bigtts", "Vivi (Female, multilingual)"),
    ("zh_female_xiaohe_uranus_bigtts", "Mindy (Female, multilingual)"),
    ("zh_male_m191_uranus_bigtts", "Kian (Male, multilingual)"),
    ("zh_male_taocheng_uranus_bigtts", "Cedric (Male, multilingual)"),
    ("zh_male_sophie_uranus_bigtts", "Sophie (Female, multilingual)"),
    ("zh_female_yingyujiaoxue_uranus_bigtts", "Jean (Female, multilingual)"),
    ("zh_male_dayi_uranus_bigtts", "Magnus (Male, multilingual)"),
    ("zh_female_mizai_uranus_bigtts", "Mabel (Female, multilingual)"),
    ("zh_female_jitangnv_uranus_bigtts", "Nadia (Female, multilingual)"),
    ("zh_female_meilinvyou_uranus_bigtts", "Opal (Female, multilingual)"),
    ("zh_female_liuchangnv_uranus_bigtts", "Pearl (Female, multilingual)"),
    ("zh_male_ruyayichen_uranus_bigtts", "Quentin (Male, multilingual)"),
    ("zh_female_vivo_uranus_bigtts", "Vienna (Female, multilingual)"),
    ("zh_male_shaonianzixin_uranus_bigtts", "Jess (Male, multilingual)"),
    ("es_male_felipe_uranus_bigtts", "Felipe (Male, Spanish)"),
    ("fr_male_usseau_uranus_bigtts", "Usseau (Male, French)"),
    ("de_male_seven_uranus_bigtts", "Sven (Male, German)"),
    ("it_male_enzo_uranus_bigtts", "Enzo (Male, Italian)"),
    ("pt_male_martins_uranus_bigtts", "Martins (Male, Portuguese)"),
    ("jp_female_minimi_uranus_bigtts", "Minimi (Female, Japanese)"),
    ("kr_male_shane_uranus_bigtts", "Shane (Male, Korean)"),
    ("id_male_han_uranus_bigtts", "Han (Male, Indonesian)"),
]
VOICE_IDS = {label: sid for sid, label in PRESET_VOICES}

_MAX_REF_SECONDS = 30.0
_MAX_REF_BYTES = 10 * 1024 * 1024
_MIN_IMAGE_PIXELS = 160_000


def _audio_to_b64_wav(audio):
    """ComfyUI AUDIO -> base64 16-bit mono WAV (small enough for the 10 MB cap)."""
    import soundfile as sf
    wf = audio["waveform"]
    sr = int(audio["sample_rate"])
    w = (wf[0] if wf.dim() == 3 else wf).float().cpu().numpy()  # [C, T]
    seconds = w.shape[-1] / sr
    if seconds > _MAX_REF_SECONDS:
        raise ValueError(f"A reference clip is {seconds:.1f}s; the limit is 30s. Trim it first.")
    buf = io.BytesIO()
    sf.write(buf, w.mean(axis=0), sr, format="WAV", subtype="PCM_16")
    data = buf.getvalue()
    if len(data) > _MAX_REF_BYTES:
        raise ValueError("A reference clip is over 10 MB; use a shorter clip.")
    return base64.b64encode(data).decode()


def _image_to_b64_png(image):
    arr = (image[0].cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    if im.width * im.height < _MIN_IMAGE_PIXELS:
        scale = (_MIN_IMAGE_PIXELS / (im.width * im.height)) ** 0.5
        im = im.resize((int(im.width * scale) + 1, int(im.height * scale) + 1), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _decode_audio(data):
    import soundfile as sf
    samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return {"waveform": torch.from_numpy(samples.T.copy()).unsqueeze(0), "sample_rate": int(sr)}


class BytePlusSeedAudio(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="BytePlusSeedAudio",
            display_name="BytePlus Seed Audio 1.0",
            category=CAT,
            description="Speech, dialogue, music and sound effects from one prompt (up to "
            "2 minutes). Describe voices, emotion, ambience and music, and write the lines "
            "to speak.",
            search_aliases=["seed audio", "byteplus", "tts", "text to speech", "voice",
                            "voice clone", "dialogue", "sound effects", "music"],
            inputs=[
                IO.String.Input("prompt", multiline=True, default="",
                                tooltip="Voices, emotion, ambience, music and the lines to speak. "
                                "With cloned voices, call them @Audio1, @Audio2, @Audio3. "
                                "Max 2048 characters."),
                IO.DynamicCombo.Input("voice", options=[
                    IO.DynamicCombo.Option(MODE_TEXT, []),
                    IO.DynamicCombo.Option(MODE_PRESET, [
                        IO.Combo.Input("preset", options=list(VOICE_IDS),
                                       default=PRESET_VOICES[0][1]),
                    ]),
                    IO.DynamicCombo.Option(MODE_CLONE, [
                        IO.Audio.Input("audio_1", tooltip="@Audio1. Up to 30 s."),
                        IO.Audio.Input("audio_2", optional=True, tooltip="@Audio2."),
                        IO.Audio.Input("audio_3", optional=True, tooltip="@Audio3."),
                    ]),
                    IO.DynamicCombo.Option(MODE_IMAGE, [
                        IO.Image.Input("image", tooltip="A character; the voice is derived "
                                       "from it."),
                    ]),
                ], tooltip="Where the voice comes from."),
                IO.Int.Input("speed", default=0, min=-50, max=100,
                             tooltip="0 normal, 100 = 2x faster, -50 = half speed."),
                IO.Int.Input("pitch", default=0, min=-12, max=12, advanced=True, optional=True,
                             tooltip="Semitones."),
                IO.Int.Input("loudness", default=0, min=-50, max=100, advanced=True,
                             optional=True),
                IO.Combo.Input("sample_rate", options=["24000", "32000", "44100", "48000",
                                                       "16000", "8000"],
                               default="48000", advanced=True, optional=True),
                IO.Combo.Input("model", options=["seed-audio-1.0", "seed-audio-1.0-multilingual"],
                               default="seed-audio-1.0", advanced=True, optional=True,
                               tooltip="multilingual: 20 languages and per-line timing like "
                               "\"[5.5s:8.0s] Wait for me!\" (if enabled on your account)."),
            ],
            outputs=[IO.Audio.Output()],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        import time
        return time.time()

    @classmethod
    def execute(cls, prompt, voice, speed, pitch=0, loudness=0, sample_rate="48000",
                model="seed-audio-1.0"):
        key = _env_lookup(_KEY_ENV)
        if not key:
            raise ValueError(
                "No Seed Audio API key. Add it in Settings (gear icon) > BytePlus > API keys "
                "(BytePlus console > Voice > activate Seed-Audio 1.0 > API key). The ModelArk "
                "ark- key does not work here.")
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Write a prompt.")
        if len(prompt) > 2048:
            raise ValueError(f"The prompt is {len(prompt)} characters; the limit is 2048.")

        mode = voice["voice"]
        references = None
        if mode == MODE_PRESET:
            references = [{"speaker": VOICE_IDS[voice["preset"]]}]
        elif mode == MODE_CLONE:
            clips = [voice.get(f"audio_{i}") for i in (1, 2, 3)]
            references = [{"audio_data": _audio_to_b64_wav(c)} for c in clips if c is not None]
            if not references:
                raise ValueError("Connect at least one audio clip to clone.")
        elif mode == MODE_IMAGE:
            if voice.get("image") is None:
                raise ValueError("Connect an image.")
            references = [{"image_data": _image_to_b64_png(voice["image"])}]

        body = {"model": model, "text_prompt": prompt, "watermark": {},
                "audio_config": {"format": "wav", "sample_rate": int(sample_rate),
                                 "speech_rate": speed, "loudness_rate": loudness,
                                 "pitch_rate": pitch}}
        if references:
            body["references"] = references

        _check_interrupt()
        r = requests.post(SEED_AUDIO_URL, json=body, timeout=180,
                          headers={"X-Api-Key": key, "Content-Type": "application/json"})
        try:
            data = r.json()
        except ValueError:
            data = {}
        if r.status_code >= 400 or (data.get("code") not in (None, 0, 20000000) and not data.get("audio")):
            hint = ""
            if "SpeakerList empty" in r.text:
                hint = " Text-only mode was rejected this time; pick a preset voice or clone one."
            raise RuntimeError(f"Seed Audio failed [{r.status_code}] "
                               f"(logid {r.headers.get('X-Tt-Logid', '-')}): "
                               f"{_redact(r.text[:600])}{hint}")
        if data.get("audio"):
            raw = base64.b64decode(data["audio"])
        elif data.get("url"):
            raw = requests.get(data["url"], timeout=120).content
        else:
            raise RuntimeError(f"Seed Audio returned no audio: {_redact(str(data)[:500])}")
        return IO.NodeOutput(_decode_audio(raw))


NODES = [BytePlusSeedAudio]
# Async, so independent BytePlus nodes in one workflow run at the same time.
NODES = [run_in_parallel(n) for n in NODES]
NODE_CLASS_MAPPINGS = {n.define_schema().node_id: n for n in NODES}
NODE_DISPLAY_NAME_MAPPINGS = {n.define_schema().node_id: n.define_schema().display_name
                              for n in NODES}
