// Narrow the BytePlus Seedance node's widgets to whatever the selected model
// actually supports.
//
// ComfyUI evaluates INPUT_TYPES() once when the node class is registered, so a
// combo cannot depend on another widget's value from Python alone. This
// extension rewrites the resolution / duration / output_format widgets whenever
// the model changes.
//
// The limits come from the node itself (/byteplus_seedance/model_caps), so they
// are defined once, in MODEL_CAPS in nodes.py. This is cosmetic only - the node
// re-validates every request server-side before a task is submitted.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "BytePlusSeedanceVideo";

let CAPS = null;

function classOf(node) {
  return node.comfyClass ?? node.constructor?.comfyClass;
}

// Mirrors _caps_for() in nodes.py: exact id first, then the model family, so a
// model id with a newer date suffix still resolves.
function capsFor(modelId) {
  if (!CAPS || !modelId) return null;
  const table = CAPS.models;
  if (table[modelId]) return table[modelId];
  const family = ["seedance-2-5", "seedance-2-0-fast", "seedance-2-0-mini",
                  "seedance-2-0"].find((f) => modelId.includes(f));
  if (!family) return null;
  const key = Object.keys(table).find((k) => k.includes(family));
  return key ? table[key] : null;
}

function applyCaps(node) {
  if (!CAPS || !node.widgets) return;
  const widget = (name) => node.widgets.find((w) => w.name === name);
  const model = widget("model");
  if (!model) return;
  const caps = capsFor(model.value);

  const resolution = widget("resolution");
  if (resolution) {
    const values = caps?.resolutions?.length ? caps.resolutions
                                             : CAPS.all_resolutions;
    resolution.options.values = values;
    if (!values.includes(resolution.value)) {
      resolution.value = values.includes("720p") ? "720p"
                                                 : values[values.length - 1];
    }
  }

  const duration = widget("duration");
  if (duration && caps?.duration?.length === 2) {
    const [lo, hi] = caps.duration;
    duration.options.max = hi;
    // -1 means "let the model choose" and is valid for every model.
    if (duration.value > hi) duration.value = hi;
    else if (duration.value !== -1 && duration.value < lo) duration.value = lo;
  }

  const outputFormat = widget("output_format");
  if (outputFormat) {
    const values = caps?.output_format ? CAPS.all_output_formats : ["mp4"];
    outputFormat.options.values = values;
    if (!values.includes(outputFormat.value)) outputFormat.value = "mp4";
  }

  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "byteplus.seedance.model_caps",

  async setup() {
    try {
      const resp = await api.fetchApi("/byteplus_seedance/model_caps");
      if (resp.ok) CAPS = await resp.json();
    } catch (err) {
      console.warn("[BytePlusSeedance] per-model widget limits unavailable:", err);
    }
    // Nodes restored from a workflow already exist by the time setup() runs.
    for (const node of app.graph?._nodes ?? []) {
      if (classOf(node) === NODE_CLASS) applyCaps(node);
    }
  },

  nodeCreated(node) {
    if (classOf(node) !== NODE_CLASS) return;
    const model = node.widgets?.find((w) => w.name === "model");
    if (!model) return;
    const previous = model.callback;
    model.callback = function (...args) {
      const result = previous?.apply(this, args);
      applyCaps(node);
      return result;
    };
    // Saved widget values are restored after nodeCreated(), so defer the first
    // pass to the next tick rather than narrowing against the defaults.
    setTimeout(() => applyCaps(node), 0);
  },
});
