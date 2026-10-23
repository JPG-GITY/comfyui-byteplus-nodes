// Settings > BytePlus: enter the API keys from inside ComfyUI.
//
// The keys never go through ComfyUI's own settings store (comfy.settings.json):
// each row talks to /byteplus/credentials (credentials.py), which saves them in
// ComfyUI's user folder (user/byteplus.env) and only ever answers with whether a
// key is set and its last four characters. Inputs are cleared after saving.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const ROUTE = "/byteplus/credentials";
const CATEGORY = ["BytePlus", "API keys"];

const SOURCE_TEXT = {
  environment: "from an environment variable",
  settings: "saved here",
  file: "from a .env file",
};

async function request(method, body) {
  const res = await api.fetchApi(ROUTE, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  let data = {};
  try {
    data = await res.json();
  } catch {
    // reported through the status code below
  }
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function el(tag, props = {}, style = {}) {
  const node = document.createElement(tag);
  Object.assign(node, props);
  Object.assign(node.style, style);
  return node;
}

const INPUT_STYLE = {
  flex: "1 1 12rem", minWidth: "0", padding: "0.35rem 0.5rem", borderRadius: "0.375rem",
  border: "1px solid var(--border-color, #4e4e4e)", background: "var(--comfy-input-bg, #222)",
  color: "var(--input-text, inherit)",
};
const BUTTON_STYLE = {
  padding: "0.35rem 0.8rem", borderRadius: "0.375rem", cursor: "pointer", flex: "0 0 auto",
  border: "1px solid var(--border-color, #4e4e4e)", background: "var(--comfy-menu-bg, #353535)",
  color: "var(--input-text, inherit)",
};

const editors = new Map();

function editor({ credential, fields }) {
  const root = el("div", {}, { display: "flex", flexDirection: "column", gap: "0.35rem",
                               width: "30rem", maxWidth: "100%" });
  const row = el("div", {}, { display: "flex", gap: "0.4rem", flexWrap: "wrap", alignItems: "center" });
  const inputs = {};
  for (const f of fields) {
    const input = el("input", { type: "password", placeholder: f.placeholder, spellcheck: false,
                                autocomplete: "new-password" }, INPUT_STYLE);
    // An API key is not a login: keep password managers out of it.
    for (const a of ["data-1p-ignore", "data-lpignore", "data-bwignore"]) input.setAttribute(a, "true");
    inputs[f.name] = input;
    row.append(input);
  }
  const save = el("button", { type: "button", textContent: "Save" }, BUTTON_STYLE);
  const remove = el("button", { type: "button", textContent: "Remove" }, BUTTON_STYLE);
  row.append(save, remove);
  const status = el("div", {}, { fontSize: "0.8rem", opacity: "0.8", minHeight: "1.1rem" });
  root.append(row, status);

  const show = (text, error = false) => {
    status.textContent = text;
    status.style.color = error ? "var(--error-text, #ff6b6b)" : "";
  };

  root.render = (data) => {
    const info = data.credentials?.[credential];
    if (!info) return;
    if (!info.configured) {
      const missing = info.entries.filter((e) => !e.configured).map((e) => e.name);
      show(`Not set (${missing.join(", ")}).`);
    } else {
      const e = info.entries[0];
      show(`Set (${info.entries.map((x) => x.hint).join(" / ")}), ${SOURCE_TEXT[e.source] ?? ""}.`);
    }
    remove.disabled = !info.removable;
    remove.title = info.removable ? "" : "Nothing saved here to remove.";
  };

  const send = async (body) => {
    save.disabled = remove.disabled = true;
    try {
      const data = await request("POST", { credential, ...body });
      for (const input of Object.values(inputs)) input.value = "";
      for (const ed of editors.values()) ed.render(data);
    } catch (err) {
      show(err.message, true);
    } finally {
      save.disabled = false;
    }
  };

  save.addEventListener("click", () => {
    const values = {};
    for (const f of fields) {
      const v = inputs[f.name].value.trim();
      if (!v) return show(`Fill in ${f.placeholder} first.`, true);
      values[f.name] = v;
    }
    send({ values });
  });
  remove.addEventListener("click", () => send({ clear: true }));

  root.refresh = async () => {
    try {
      root.render(await request("GET"));
    } catch (err) {
      show(`Could not read the key status: ${err.message}`, true);
    }
  };
  root.refresh();
  return root;
}

// ComfyUI re-runs a custom setting's render function on every re-render, so
// each editor is built once and reused.
function editorFor(id, options) {
  let ed = editors.get(id);
  if (!ed) {
    ed = editor(options);
    editors.set(id, ed);
  } else {
    ed.refresh();
  }
  return ed;
}

function setting(id, name, tooltip, options) {
  return {
    id: `BytePlus.Keys.${id}`,
    name,
    category: [...CATEGORY, name],
    tooltip,
    type: () => editorFor(id, options),
    defaultValue: "", // never used: nothing is stored in ComfyUI's settings
  };
}

app.registerExtension({
  name: "byteplus.settings.keys",
  settings: [
    setting("ModelArk", "ModelArk API key",
      "Seedance and Seedream nodes. ModelArk console > API keys (starts with ark-).",
      { credential: "modelark", fields: [{ name: "BYTEPLUS_API_KEY", placeholder: "ark-..." }] }),
    setting("SeedAudio", "Seed Audio API key",
      "Seed Audio 1.0 node. BytePlus console > Voice > activate Seed-Audio 1.0 > API key.",
      { credential: "seed_audio",
        fields: [{ name: "BYTEPLUS_SEED_AUDIO_API_KEY", placeholder: "Seed Audio API key" }] }),
    setting("MediaKit", "AI MediaKit API key",
      "All MediaKit nodes. AI MediaKit console > Settings > API key.",
      { credential: "mediakit",
        fields: [{ name: "BYTEPLUS_MEDIAKIT_API_KEY", placeholder: "MediaKit API key" }] }),
    setting("Storage", "Access key / Secret key (TOS)",
      "Needed to send connected videos, audio and MediaKit inputs (uploaded to your TOS " +
      "bucket), and for the asset library. IAM > Access keys.",
      { credential: "storage",
        fields: [{ name: "ARK_ACCESS_KEY", placeholder: "Access key (AK)" },
                 { name: "ARK_SECRET_KEY", placeholder: "Secret key (SK)" }] }),
    setting("SessionToken", "Session token (temporary keys only)",
      "Only for temporary (STS) keys, whose access key starts with AKTP: paste the session " +
      "token that came with them. It expires, so update it (and the AK/SK) when it does. " +
      "Leave empty for permanent IAM keys (AKLT...).",
      { credential: "session_token",
        fields: [{ name: "ARK_SESSION_TOKEN", placeholder: "Session token" }] }),
  ],
});
