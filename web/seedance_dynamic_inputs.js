// Grow the reference sockets on demand instead of showing all of them at once.
//
// Seedance 2.5 accepts 30 images / 10 videos / 10 audio clips, so declaring
// every slot up front puts ~31 sockets on the node and makes it unusable. The
// node still DECLARES them all (Python needs the names), but this extension
// keeps only the connected ones plus a single spare per kind visible: connect
// ref_image_1 and ref_image_2 appears, and so on.
//
// Shrinking only ever removes unconnected sockets, so a link is never cut. The
// backend reads its sockets by name (`**ref_slots`), so a socket that is not on
// the node simply arrives as None — nothing to keep in sync server-side.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "BytePlusSeedanceVideo";

// Reference families, in the order the node sends them to the API. `max` is how
// many sockets of that kind the node declares — 30 / 10 / 10, one per reference
// Seedance 2.5 accepts, 50 in all. These defaults mirror _IMAGE_SLOTS etc. in
// nodes.py and are refreshed from the node itself at startup (see setup), so
// changing the Python side alone cannot leave this file growing into sockets
// that do not exist.
const FAMILIES = [
  { kind: "image", prefix: "ref_image_", max: 30, type: "IMAGE" },
  { kind: "video", prefix: "ref_video_", max: 10, type: "VIDEO" },
  { kind: "audio", prefix: "ref_audio_", max: 10, type: "AUDIO" },
];
// Frame mode is two sockets and mutually exclusive with reference mode, so it is
// always shown. These lead the input list, matching INPUT_TYPES in nodes.py.
const STATIC_INPUTS = ["first_frame", "last_frame"];

const familyNames = (f) =>
  Array.from({ length: f.max }, (_, i) => `${f.prefix}${i + 1}`);

const familyFor = (name) => FAMILIES.find((f) => name.startsWith(f.prefix));

function classOf(node) {
  return node.comfyClass ?? node.constructor?.comfyClass;
}

// The input names this node should currently expose, in canonical order.
function computeDesired(node) {
  const inputs = node.inputs || [];
  const connected = new Set(
    inputs.filter((i) => i.link != null).map((i) => i.name)
  );

  const managed = new Set(STATIC_INPUTS);
  for (const f of FAMILIES) for (const n of familyNames(f)) managed.add(n);

  const desired = [...STATIC_INPUTS];
  for (const f of FAMILIES) {
    const all = familyNames(f);
    let lastConnected = 0; // 1-based; 0 = none connected in this family
    all.forEach((n, i) => {
      if (connected.has(n)) lastConnected = i + 1;
    });
    // Everything up to the last connected socket, plus one spare to grow into.
    desired.push(...all.slice(0, Math.min(lastConnected + 1, f.max)));
  }

  // Anything we don't own (e.g. a widget the user converted to an input) keeps
  // its place at the end rather than being dropped.
  for (const inp of inputs) if (!managed.has(inp.name)) desired.push(inp.name);
  return desired;
}

function applyDynamicInputs(node) {
  if (!node || !node.inputs || node.__seedanceSyncing) return;
  node.__seedanceSyncing = true;
  try {
    const desired = computeDesired(node);
    const present = new Set(node.inputs.map((i) => i.name));

    // Add the newly-needed spares. Never invent an input we don't own.
    for (const name of desired) {
      if (present.has(name)) continue;
      const f = familyFor(name);
      if (f) node.addInput(name, f.type);
    }

    // Drop surplus sockets, tail first. A connected socket is never removed, so
    // this can't cut a link even if the desired set is computed wrongly.
    const wanted = new Set(desired);
    for (let i = node.inputs.length - 1; i >= 0; i--) {
      const inp = node.inputs[i];
      if (wanted.has(inp.name) || inp.link != null) continue;
      node.removeInput(i);
    }

    // addInput() appends, so restore canonical order and re-point every link at
    // its new slot index (litegraph stores that index on the link itself).
    const byName = new Map(node.inputs.map((i) => [i.name, i]));
    const ordered = desired.map((n) => byName.get(n)).filter(Boolean);
    if (ordered.length === node.inputs.length) {
      node.inputs = ordered;
      node.inputs.forEach((inp, idx) => {
        if (inp.link == null) return;
        const link = node.graph?.links?.[inp.link];
        if (link) link.target_slot = idx;
      });
    }

    // Keep the user's width; only the height should follow the socket count.
    if (node.computeSize && node.setSize) {
      node.setSize([node.size?.[0] ?? node.computeSize()[0], node.computeSize()[1]]);
    }
    node.setDirtyCanvas?.(true, true);
  } finally {
    node.__seedanceSyncing = false;
  }
}

app.registerExtension({
  name: "byteplus.seedance.dynamic_inputs",

  // Take the real socket counts from the node, so the two sides cannot drift.
  // Failure is harmless: the defaults above already match nodes.py.
  async setup() {
    try {
      const resp = await api.fetchApi("/byteplus_seedance/model_caps");
      if (resp.ok) {
        const slots = (await resp.json()).slots || {};
        for (const f of FAMILIES) {
          const n = Number(slots[f.kind]);
          if (Number.isInteger(n) && n > 0) f.max = n;
        }
      }
    } catch (err) {
      console.warn("[BytePlusSeedance] socket counts unavailable, using defaults:", err);
    }
    for (const node of app.graph?._nodes ?? []) {
      if (classOf(node) === NODE_CLASS) applyDynamicInputs(node);
    }
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_CLASS) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      // Collapse on the next tick, once ComfyUI has finished building the
      // default (full) input list.
      setTimeout(() => applyDynamicInputs(this), 0);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      // Loading a workflow restores links by slot index during configure, so
      // only re-evaluate after that has finished — never during.
      setTimeout(() => applyDynamicInputs(this), 0);
      return r;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
      const r = onConnectionsChange?.apply(this, arguments);
      const INPUT = globalThis.LiteGraph?.INPUT ?? 1;
      if (type === INPUT && classOf(this) === NODE_CLASS) {
        setTimeout(() => applyDynamicInputs(this), 0);
      }
      return r;
    };
  },
});
