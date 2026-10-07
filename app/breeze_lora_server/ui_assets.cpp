#include "ui_assets.h"

namespace breeze_lora_server {

std::string_view embedded_ui_html() noexcept {
    return R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Breeze LoRA</title>
<style>
:root {
  --bg: #12141a;
  --panel: #1a1d26;
  --line: #2c3140;
  --text: #e8e6e3;
  --muted: #9a9aa3;
  --accent: #c4a574;
  --accent-dim: #8a7349;
  --danger: #c45c5c;
  --ok: #6a9a7a;
  --focus: #d4b888;
  --radius: 6px;
  --font: "Segoe UI", "Helvetica Neue", sans-serif;
  --mono: "Cascadia Code", "Consolas", monospace;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0; min-height: 100%;
  background: radial-gradient(1200px 600px at 10% -10%, #1e2330 0%, var(--bg) 55%);
  color: var(--text); font-family: var(--font);
}
body { padding: 1.5rem clamp(1rem, 3vw, 2.5rem) 3rem; }
header {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.75rem 1.25rem;
  margin-bottom: 1.5rem; border-bottom: 1px solid var(--line); padding-bottom: 1rem;
}
h1 {
  margin: 0; font-size: 1.35rem; font-weight: 600; letter-spacing: 0.02em;
}
.tag {
  color: var(--muted); font-size: 0.85rem;
}
main {
  display: grid; gap: 1.25rem;
  grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
}
@media (max-width: 900px) {
  main { grid-template-columns: 1fr; }
}
section {
  background: color-mix(in srgb, var(--panel) 92%, black);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 1rem 1.1rem 1.15rem;
}
section h2 {
  margin: 0 0 0.85rem; font-size: 0.78rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted);
}
label {
  display: block; font-size: 0.8rem; color: var(--muted); margin: 0.7rem 0 0.3rem;
}
label:first-of-type { margin-top: 0; }
select, textarea, input[type="file"] {
  width: 100%; background: #0f1117; color: var(--text);
  border: 1px solid var(--line); border-radius: var(--radius);
  padding: 0.55rem 0.65rem; font: inherit;
}
select:focus, textarea:focus {
  outline: 1px solid var(--focus); border-color: var(--accent-dim);
}
textarea { min-height: 5.5rem; resize: vertical; line-height: 1.4; }
textarea.short { min-height: 3.2rem; }
.row {
  display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 0.9rem; align-items: center;
}
button {
  appearance: none; border: 1px solid var(--line); background: #232734; color: var(--text);
  border-radius: var(--radius); padding: 0.5rem 0.85rem; font: inherit; cursor: pointer;
}
button.primary {
  background: linear-gradient(180deg, #d0b07e, var(--accent));
  color: #1a140c; border-color: var(--accent-dim); font-weight: 600;
}
button.danger { border-color: #6e3a3a; color: #f0c8c8; }
button:disabled {
  opacity: 0.45; cursor: not-allowed;
}
button:not(:disabled):hover { filter: brightness(1.06); }
.pill {
  display: inline-flex; align-items: center; gap: 0.35rem;
  font-size: 0.78rem; color: var(--muted);
  border: 1px solid var(--line); border-radius: 999px; padding: 0.2rem 0.65rem;
}
.pill.on { color: var(--ok); border-color: #3d5a46; }
.pill.off { color: var(--muted); }
#status {
  min-height: 1.35rem; margin-top: 0.85rem; font-size: 0.85rem; color: var(--muted);
  white-space: pre-wrap; word-break: break-word;
}
#status.error { color: var(--danger); }
#status.ok { color: var(--ok); }
.history-item {
  border-top: 1px solid var(--line); padding: 0.85rem 0 0.2rem; margin-top: 0.75rem;
}
.history-item:first-child { border-top: none; margin-top: 0; padding-top: 0; }
.history-meta {
  font-size: 0.78rem; color: var(--muted); margin-bottom: 0.45rem;
  font-family: var(--mono);
}
.history-item audio { width: 100%; margin: 0.35rem 0; }
.history-item a { color: var(--accent); font-size: 0.85rem; }
.hint { font-size: 0.78rem; color: var(--muted); margin-top: 0.45rem; }
</style>
</head>
<body>
<header>
  <h1>Breeze LoRA</h1>
  <span class="tag">audition · instruction · reference preset</span>
  <span id="mgmtPill" class="pill off">management: checking…</span>
</header>
<main>
  <section>
    <h2>Speak</h2>
    <label for="model">Model</label>
    <select id="model"></select>
    <label for="input">Input</label>
    <textarea id="input" placeholder="Text to speak"></textarea>
    <label for="instruction">Instruction</label>
    <textarea id="instruction" class="short" placeholder="Speaking style instruction"></textarea>
    <label for="referenceAudio">Reference WAV (optional, unsaved overrides preset)</label>
    <input id="referenceAudio" type="file" accept="audio/wav,.wav">
    <label for="referenceText">Reference transcript</label>
    <textarea id="referenceText" class="short" placeholder="Transcript of the reference WAV"></textarea>
    <div class="row">
      <span id="refPill" class="pill off">reference: none</span>
    </div>
    <div class="row">
      <button id="generateBtn" class="primary" type="button">Generate</button>
      <button id="saveBtn" type="button">Save preset</button>
      <button id="clearBtn" class="danger" type="button">Clear reference</button>
    </div>
    <p class="hint">Unsaved WAV/transcript overrides the saved model reference. Unsaved instruction overrides the saved default. Instruction-only presets are valid.</p>
    <div id="status"></div>
  </section>
  <section>
    <h2>Results</h2>
    <div id="history"></div>
  </section>
</main>
<script>
(() => {
  const els = {
    model: document.getElementById('model'),
    input: document.getElementById('input'),
    instruction: document.getElementById('instruction'),
    referenceAudio: document.getElementById('referenceAudio'),
    referenceText: document.getElementById('referenceText'),
    generateBtn: document.getElementById('generateBtn'),
    saveBtn: document.getElementById('saveBtn'),
    clearBtn: document.getElementById('clearBtn'),
    status: document.getElementById('status'),
    history: document.getElementById('history'),
    mgmtPill: document.getElementById('mgmtPill'),
    refPill: document.getElementById('refPill'),
  };

  let models = [];
  let managementEnabled = false;
  const objectUrls = [];

  function setStatus(message, kind) {
    els.status.textContent = message || '';
    els.status.className = kind || '';
  }

  async function readError(response) {
    const text = await response.text();
    try {
      const json = JSON.parse(text);
      if (json && json.error && json.error.message) return json.error.message;
      if (json && json.message) return json.message;
    } catch (_) {}
    return text || (response.status + ' ' + response.statusText);
  }

  function selectedModel() {
    return models.find((m) => m.id === els.model.value) || null;
  }

  function updateRefPill(model) {
    if (!model) {
      els.refPill.textContent = 'reference: none';
      els.refPill.className = 'pill off';
      return;
    }
    if (model.has_voice_ref) {
      els.refPill.textContent = 'reference: configured';
      els.refPill.className = 'pill on';
    } else {
      els.refPill.textContent = 'reference: none';
      els.refPill.className = 'pill off';
    }
  }

  function applyModelFields(model) {
    if (!model) return;
    els.instruction.value = model.default_instruction || '';
    els.referenceText.value = model.reference_text || '';
    els.referenceAudio.value = '';
    updateRefPill(model);
  }

  function setManagementEnabled(enabled) {
    managementEnabled = !!enabled;
    els.saveBtn.disabled = !managementEnabled;
    els.clearBtn.disabled = !managementEnabled;
    els.referenceAudio.disabled = !managementEnabled;
    els.mgmtPill.textContent = managementEnabled
      ? 'management: enabled'
      : 'management: loopback only (audition still works)';
    els.mgmtPill.className = managementEnabled ? 'pill on' : 'pill off';
  }

  async function loadModels() {
    setStatus('Loading models…');
    const response = await fetch('/ui/models');
    if (!response.ok) throw new Error(await readError(response));
    const data = await response.json();
    models = Array.isArray(data.models) ? data.models : [];
    setManagementEnabled(!!data.management_enabled);
    const previous = els.model.value;
    els.model.innerHTML = '';
    for (const model of models) {
      const opt = document.createElement('option');
      opt.value = model.id;
      opt.textContent = model.id;
      els.model.appendChild(opt);
    }
    if (models.length === 0) {
      applyModelFields(null);
      setStatus('No models configured', 'error');
      return;
    }
    const pick = models.find((m) => m.id === previous) || models[0];
    els.model.value = pick.id;
    applyModelFields(pick);
    setStatus('Ready', 'ok');
  }

  function requireInput() {
    const text = els.input.value.trim();
    if (!text) throw new Error('Input text is required');
    return text;
  }

  function pickedWav() {
    return els.referenceAudio.files && els.referenceAudio.files[0]
      ? els.referenceAudio.files[0]
      : null;
  }

  async function savePreset() {
    if (!managementEnabled) throw new Error('Management UI requires a loopback bind');
    const model = selectedModel();
    if (!model) throw new Error('Select a model');
    const instruction = els.instruction.value.trim();
    if (!instruction) throw new Error('Instruction must be non-empty');
    const wav = pickedWav();
    const transcript = els.referenceText.value.trim();
    if (wav && !transcript) throw new Error('Reference transcript is required when a WAV is selected');

    setStatus('Saving preset…');
    let response;
    if (wav) {
      const form = new FormData();
      form.append('reference_audio', wav, wav.name || 'reference.wav');
      form.append('reference_text', transcript);
      form.append('default_instruction', instruction);
      response = await fetch('/ui/models/' + encodeURIComponent(model.id) + '/reference', {
        method: 'POST',
        body: form,
      });
    } else {
      response = await fetch('/ui/models/' + encodeURIComponent(model.id), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          default_instruction: instruction,
          reference_text: transcript ? transcript : null,
          clear_reference: false,
        }),
      });
    }
    if (!response.ok) throw new Error(await readError(response));
    await loadModels();
    setStatus('Preset saved', 'ok');
  }

  async function clearReference() {
    if (!managementEnabled) throw new Error('Management UI requires a loopback bind');
    const model = selectedModel();
    if (!model) throw new Error('Select a model');
    const instruction = els.instruction.value.trim();
    if (!instruction) throw new Error('Instruction must be non-empty');
    setStatus('Clearing reference…');
    const response = await fetch('/ui/models/' + encodeURIComponent(model.id), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        default_instruction: instruction,
        reference_text: null,
        clear_reference: true,
      }),
    });
    if (!response.ok) throw new Error(await readError(response));
    await loadModels();
    setStatus('Reference cleared', 'ok');
  }

  function addHistory(blob, meta) {
    const url = URL.createObjectURL(blob);
    objectUrls.push(url);
    const item = document.createElement('div');
    item.className = 'history-item';
    const metaEl = document.createElement('div');
    metaEl.className = 'history-meta';
    metaEl.textContent = meta;
    const audio = document.createElement('audio');
    audio.controls = true;
    audio.src = url;
    const link = document.createElement('a');
    link.href = url;
    link.download = 'breeze-' + Date.now() + '.wav';
    link.textContent = 'Download WAV';
    item.appendChild(metaEl);
    item.appendChild(audio);
    item.appendChild(link);
    els.history.prepend(item);
  }

  async function generate() {
    const model = selectedModel();
    if (!model) throw new Error('Select a model');
    const input = requireInput();
    const instruction = els.instruction.value.trim();
    if (!instruction) throw new Error('Instruction must be non-empty');
    const wav = pickedWav();
    const transcript = els.referenceText.value.trim();
    if (wav && !transcript) throw new Error('Reference transcript is required when a WAV is selected');

    setStatus('Generating…');
    let response;
    if (wav) {
      const form = new FormData();
      form.append('model', model.id);
      form.append('input', input);
      form.append('instruction', instruction);
      form.append('reference_audio', wav, wav.name || 'reference.wav');
      form.append('reference_text', transcript);
      response = await fetch('/ui/audio/speech', { method: 'POST', body: form });
    } else {
      response = await fetch('/ui/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: model.id,
          input,
          instruction,
          response_format: 'wav',
        }),
      });
    }
    if (!response.ok) throw new Error(await readError(response));
    const blob = await response.blob();
    addHistory(blob, model.id + ' · ' + new Date().toLocaleTimeString());
    setStatus('Generated', 'ok');
  }

  function wrap(fn) {
    return async () => {
      try {
        els.generateBtn.disabled = true;
        els.saveBtn.disabled = true;
        els.clearBtn.disabled = true;
        await fn();
      } catch (err) {
        setStatus(err && err.message ? err.message : String(err), 'error');
      } finally {
        els.generateBtn.disabled = false;
        setManagementEnabled(managementEnabled);
      }
    };
  }

  els.model.addEventListener('change', () => applyModelFields(selectedModel()));
  els.generateBtn.addEventListener('click', wrap(generate));
  els.saveBtn.addEventListener('click', wrap(savePreset));
  els.clearBtn.addEventListener('click', wrap(clearReference));
  window.addEventListener('beforeunload', () => {
    for (const url of objectUrls) URL.revokeObjectURL(url);
  });

  loadModels().catch((err) => {
    setStatus(err && err.message ? err.message : String(err), 'error');
  });
})();
</script>
</body>
</html>
)HTML";
}

}  // namespace breeze_lora_server
