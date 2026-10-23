// BytePlus AI MediaKit node: load the selected tool's request template into
// params_json, and add buttons to reset it or open the tool's API reference.
//
// Templates and docs come from the node itself (/byteplus_mediakit/tools), so
// the tool catalog lives in one place: mediakit_tools.json.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "BytePlusMediaKit";

let TOOLS = null;

const pretty = (obj) => JSON.stringify(obj ?? {}, null, 2);

function classOf(node) {
  return node.comfyClass ?? node.constructor?.comfyClass;
}

function setup(node) {
  const widget = (name) => node.widgets?.find((w) => w.name === name);
  const tool = widget("tool");
  const params = widget("params_json");
  if (!tool || !params) return;

  const templateFor = (label) => TOOLS?.[label] ? pretty(TOOLS[label].template) : null;
  let lastTemplate = templateFor(tool.value);

  const describe = () => {
    const info = TOOLS?.[tool.value];
    if (!info) return;
    const req = info.params.filter((p) => p.required).map((p) => p.name);
    tool.tooltip = `${info.title}${info.sync ? " (sync)" : " (async task)"}\n\n` +
      `${info.summary}\n\nRequired: ${req.join(", ") || "none"}\nDocs: ${info.doc}`;
    params.tooltip = info.params
      .map((p) => `${p.required ? "* " : ""}${p.name} (${p.type}): ${p.description}`)
      .join("\n\n");
  };

  const original = tool.callback;
  tool.callback = function (value, ...rest) {
    const result = original?.call(this, value, ...rest);
    const next = templateFor(tool.value);
    // Only replace params the user has not edited away from the old template.
    const untouched = !params.value?.trim() || params.value === lastTemplate;
    if (next && untouched) params.value = next;
    lastTemplate = next;
    describe();
    node.setDirtyCanvas(true, true);
    return result;
  };

  node.addWidget("button", "reset params to tool template", null, () => {
    const next = templateFor(tool.value);
    if (next) {
      params.value = next;
      lastTemplate = next;
      node.setDirtyCanvas(true, true);
    }
  });
  node.addWidget("button", "open tool API docs", null, () => {
    const info = TOOLS?.[tool.value];
    if (info) window.open(info.doc, "_blank", "noopener");
  });
  describe();
}

app.registerExtension({
  name: "byteplus.mediakit.tools",
  async setup() {
    try {
      const res = await api.fetchApi("/byteplus_mediakit/tools");
      TOOLS = (await res.json()).tools;
    } catch (err) {
      console.warn("[BytePlusMediaKit] could not load tool templates", err);
    }
  },
  nodeCreated(node) {
    if (classOf(node) === NODE_CLASS) setup(node);
  },
});
