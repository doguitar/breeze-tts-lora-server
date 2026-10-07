# breeze_lora_server

Dedicated OpenAI-compatible Breeze TTS 2 server with Instavar LoRA hot-swap.

This repository **serves** adapters. It does **not** train them. Training and
export stay in Python; the C++ binary loads a resident Breeze GGUF base exposed
as the fixed OpenAI model `breeze-base`, and selects Instavar LoRA / instruction
presets through the request `voice` field.

## Creating a LoRA for this server

### 1. Train outside this repo

Official [`breezeblue-ai/breeze-tts`](https://github.com/breezeblue-ai/breeze-tts)
is inference-only. Use an independent training toolkit that exports Instavar-
compatible adapters, for example:

- Toolkit: [`instavar/breeze-tts2-finetuning`](https://github.com/instavar/breeze-tts2-finetuning)
- Experiment notes: [Instavar LoRA / full-SFT writeup](https://instavar.com/research/tts/breeze-tts-2-lora-full-sft-singapore-english)

Typical flow:

1. Start from the pinned Breeze TTS 2 checkpoint you will also serve as GGUF.
2. Prepare paired audio + transcript (+ optional instruction) data.
3. Train **LoRA** (recommended for a single ~24 GB GPU) with Instavar defaults.
4. Export an adapter **directory** (not a merged full checkpoint).

Full SFT is heavier and is not required for this server. The server expects a
side-adapter package, not a permanently merged weight dump.

### 2. Export the layout this server accepts (v1)

Each adapter directory must contain at least:

| File | Role |
|------|------|
| `adapter_config.json` | Manifest (schema below) |
| `adapter.safetensors` | LoRA `A`/`B` tensors |

**Fixed v1 constraints** (startup rejects anything else):

| Field | Required value |
|-------|----------------|
| Variant | `backbone_depth_projection` |
| Rank | `8` |
| Alpha | positive finite (Instavar default `16`; scale = `alpha / rank`) |
| Artifact type | `breeze_lora_adapter` |
| Schema | `schema_version: 1` |

**Target modules** (keys in `adapter.safetensors`):

- Backbone attention / MLP projections under `backbone_model.layers.*`:
  - `.self_attn.{q,k,v,o}_proj`
  - `.mlp.{gate,up,down}_proj`
- Depth decoder attention / MLP projections under `depth_decoder.model.layers.*`
  (same suffixes)
- Projections:
  - `text_encoder_proj`
  - `depth_decoder.model.inputs_embeds_projector`
  - `lm_head`

Tensor names must be `{module}.lora_A` and `{module}.lora_B` and map to base
weights `{module}.weight`. Missing pairs, unexpected tensors, wrong shapes, or
checksum mismatches fail at startup.

**Example `adapter_config.json`:**

```json
{
  "schema_version": 1,
  "artifact_type": "breeze_lora_adapter",
  "base_model": {
    "id": "BreezeBlue/Breeze-TTS-2",
    "revision": "799624c0b4a1daa8db6d28bbd9850043c0270734",
    "files": {
      "config.json": "<64-char-sha256-of-config.json>"
    }
  },
  "adapter": {
    "file": "adapter.safetensors",
    "sha256": "<64-char-sha256-of-adapter.safetensors>"
  },
  "lora": {
    "variant": "backbone_depth_projection",
    "rank": 8,
    "alpha": 16,
    "seed": 7
  }
}
```

Pin `base_model.revision` (and fileset hashes) to the same base you convert to
GGUF and put in `server.json` as `base_revision`. A mismatch causes startup
failure when configured.

### 3. License reminder

| Artifact | License |
|----------|---------|
| Inference / server source | Apache-2.0 (upstream audio.cpp lineage) |
| Breeze TTS 2 weights, adapters, self-hosted derivatives | [BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE) |

Fine-tunes and LoRAs are Derivative Models under that agreement. Research /
non-commercial use only unless you obtain a separate commercial license from
BreezeBlue / RESONIA.

### 4. Point the server at your voices

Example `server.json`:

```json
{
  "host": "0.0.0.0",
  "port": 8080,
  "backend": "cuda",
  "device": 0,
  "threads": 1,
  "max_queue_depth": 8,
  "base_model": "/models/Breeze-TTS-2-GGUF",
  "base_revision": "799624c0b4a1daa8db6d28bbd9850043c0270734",
  "voices": [
    {
      "id": "my-voice",
      "lora": "/loras/my-voice",
      "default_instruction": "Speak clearly and naturally.",
      "voice_ref": "/loras/my-voice/reference.wav",
      "reference_text_file": "/loras/my-voice/reference.txt"
    },
    {
      "id": "base-narrator",
      "lora": null,
      "default_instruction": "Speak clearly and naturally."
    }
  ]
}
```

Per-voice optional fields:

| Field | Role |
|------|------|
| `default_instruction` | Default Breeze instruction when the request omits `instruction` |
| `voice_ref` | Path to a reference `.wav` used for clone-mode synthesis |
| `reference_text` | Inline transcript of `voice_ref` |
| `reference_text_file` | Alternative to `reference_text`: load transcript from a `.txt` file |

Rules:

- `voices` is a JSON array (may be empty for base-only deployments). The legacy
  top-level `models` key is rejected.
- A voice may omit `lora` or set `"lora": null` for an instruction/reference-only
  base voice. A non-null `lora` must be a directory that contains a valid
  Instavar package.
- Each voice id must be unique and safe (no `/`, `\`, or `..`).
- If `voice_ref` is set, you must also set exactly one of `reference_text` or
  `reference_text_file`. Startup fails if the pair is incomplete or the files
  are missing.
- When a voice has `voice_ref`, speech requests for that voice automatically use
  Breeze clone mode with the packaged reference. Request-level `voice_ref` +
  `reference_text` override the configured default.
- Config paths (`lora`, `voice_ref`, `reference_text_file`, `base_model`) are
  resolved relative to the directory containing `server.json`. Prefer mounting
  the config beside the model tree (for example `/models/server.json`) so
  relatives like `loras/my-voice/reference.wav` resolve inside the same mount.
- Request-level `voice_ref` paths are also resolved relative to that config
  directory when they are not absolute.
- Invalid adapters / voice refs **fail startup**; they are not skipped.

Docker (CUDA):

```bash
docker build -f .devops/breeze_lora_cuda.Dockerfile -t local/breeze-lora-server:cuda12 .
# Read-only mounts are fine for API-only use. For the management WebUI, mount the
# config directory read-write so `server.json` and `webui-references/` can be saved.
docker run --gpus all -p 8080:8080 \
  -v /path/to/models:/models:ro \
  -v /path/to/loras:/loras:ro \
  -v /path/to/config-dir:/config:rw \
  local/breeze-lora-server:cuda12 --config /config/server.json
```

Bind `host` to `127.0.0.1`, `localhost`, or `::1` to enable WebUI management
(instruction/reference saves and uploads). Non-loopback binds still serve the
audition page and synthesis, but reject management writes with `403`.

`model` must be `breeze-base`. Select a configured voice with the optional
`voice` field:

```bash
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{"model":"breeze-base","voice":"my-voice","input":"The train arrives in five minutes.","seed":42}'
```

```bash
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o out.mp3 \
  -d '{"model":"breeze-base","voice":"my-voice","input":"The train arrives in five minutes.","seed":42,"response_format":"mp3"}'
```

Omit `voice` for the unadapted base (built-in instruction, no saved reference):

```bash
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{"model":"breeze-base","input":"The train arrives in five minutes.","seed":42}'
```

Accepted non-streaming `response_format` values: `wav` (default, returns
`audio/wav`) and `mp3` (returns `audio/mpeg`). `stream` / `stream_format` and any
other `response_format` are rejected with `400`.

## Endpoints

- `GET /health`
- `GET /v1/models` — singleton `{ id: "breeze-base", object: "model", owned_by: "breeze-lora-server" }`
- `POST /v1/audio/speech` (complete `audio/wav` by default, or `audio/mpeg` with `response_format=mp3`); requires `model: "breeze-base"` and accepts optional `voice`
- `GET /` — embedded WebUI (audition + management when loopback-bound)
- `GET /ui/voices` — `{ management_enabled, voices: [{ id, default_instruction, has_voice_ref, reference_text }] }`
- `PUT /ui/voices/<id>` — update `default_instruction` / clear or edit reference transcript (loopback only)
- `POST /ui/voices/<id>/reference` — multipart `reference_audio` + `reference_text` (+ optional `default_instruction`); writes `webui-references/<id>.wav` (loopback only)
- `POST /ui/audio/speech` — browser synthesis (JSON or multipart); forces WAV for multipart; request-level instruction/reference override saved defaults without persisting

The OpenAI model identity is always `breeze-base`. Configured adapters and
instruction/reference presets are selected with `voice`. Management endpoints
require a writable `server.json` (and `webui-references/` under the config
directory for uploads).

## Request logging

Each successful `/v1/audio/speech` job logs one timing line (request text is not
included on success). Logging is enabled by default to stdout; pass
`--log-file path` to append to a file.

Example:

```text
[info] [breeze_lora_server] speech request model=breeze-base voice=<id|(none)> first_load=<true|false> load_ms=<ms> generate_ms=<ms> status=ok
```

- `first_load=true` on the first live activation of that activation id in the
  process (`breeze-base` is activated at startup, so its first request is not a
  first load).
- `load_ms` is the adapter/base switch time; switching only rebinds side-adapter
  buffers, so this is normally `0.0`, and it is `0.0` when the requested voice is
  already active.
- Synthesis failures also emit `status=error` with the same timing fields.
- Any HTTP response other than `200` additionally logs full request details
  (method, path, query, headers, body) plus the response status and body.

## Config

See `example.server.json` and `base-only.server.json`. Paths are resolved
relative to the config file unless absolute.

## Docker

CPU smoke build + unit tests (one invocation; do not run parallel `docker build` jobs):

```powershell
powershell -ExecutionPolicy Bypass -File .devops/verify_cpu_docker.ps1
# Then read $env:TEMP\breeze-lora-cpu-verify-*.exit for DOCKER_EXIT and VERIFY_TAG
```

Base-only speech smoke (no LoRA) against a mounted GGUF package:

```powershell
docker run --rm -p 8080:8080 `
  -v '<HOST_MODEL_DIR>\Breeze-TTS-2-GGUF:/models/Breeze-TTS-2-GGUF:ro' `
  -v $PWD\app\breeze_lora_server\base-only.server.json:/app/server.json:ro `
  local/breeze-lora-server:cpu-verify-TAG

powershell -ExecutionPolicy Bypass -File scripts\smoke_breeze_base.ps1
```
