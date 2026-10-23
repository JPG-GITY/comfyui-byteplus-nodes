from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from . import credentials  # noqa: F401  (registers the Settings > BytePlus routes)
from .mediakit import (
    NODE_CLASS_MAPPINGS as _MEDIAKIT_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _MEDIAKIT_NAMES,
)

NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **_MEDIAKIT_CLASSES}
NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **_MEDIAKIT_NAMES}

# One simple node per common MediaKit task. They use ComfyUI's newer node API
# (comfy_api.latest); on an older ComfyUI only the nodes above load.
try:
    from .seedance_nodes import (
        NODE_CLASS_MAPPINGS as _SEEDANCE_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _SEEDANCE_NAMES,
    )
    from .seedream_nodes import (
        NODE_CLASS_MAPPINGS as _SEEDREAM_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _SEEDREAM_NAMES,
    )
    from .seedaudio_nodes import (
        NODE_CLASS_MAPPINGS as _SEEDAUDIO_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _SEEDAUDIO_NAMES,
    )
    from .mediakit_nodes import (
        NODE_CLASS_MAPPINGS as _SIMPLE_CLASSES,
        NODE_DISPLAY_NAME_MAPPINGS as _SIMPLE_NAMES,
    )
    for _classes, _names in ((_SEEDANCE_CLASSES, _SEEDANCE_NAMES),
                             (_SEEDREAM_CLASSES, _SEEDREAM_NAMES),
                             (_SEEDAUDIO_CLASSES, _SEEDAUDIO_NAMES),
                             (_SIMPLE_CLASSES, _SIMPLE_NAMES)):
        NODE_CLASS_MAPPINGS.update(_classes)
        NODE_DISPLAY_NAME_MAPPINGS.update(_names)
except ImportError as exc:
    print(f"[BytePlus] The simple Seedance / Seedream / Seed Audio / MediaKit nodes need a "
          f"newer ComfyUI ({exc}); the All-in-One and All Tools nodes still work.")

# Serves web/seedance_model_caps.js, which narrows the resolution / duration /
# output_format widgets to whatever the selected model supports, and
# web/mediakit_tools.js, which loads each MediaKit tool's request template.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
