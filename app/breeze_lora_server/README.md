# breeze_lora_server

Dedicated OpenAI-compatible Breeze TTS 2 server with Instavar LoRA hot-swap.

This repository **serves** adapters. It does **not** train them. Training and
export stay in Python; the C++ binary loads a resident Breeze GGUF base and
switches Instavar-format LoRA adapters by OpenAI `model` id.

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

### 4. Point the server at your adapters

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
  "models": [
    {
      "id": "breeze-base",
      "lora": null,
      "default_instruction": "Speak clearly and naturally."
    },
    {
      "id": "my-voice",
      "lora": "/loras/my-voice",
      "default_instruction": "Speak clearly and naturally.",
      "voice_ref": "/loras/my-voice/reference.wav",
      "reference_text_file": "/loras/my-voice/reference.txt"
    }
  ]
}
```

Per-model optional fields:

| Field | Role |
|------|------|
| `default_instruction` | Default Breeze instruction when the request omits `instruction` |
| `voice_ref` | Path to a reference `.wav` used for clone-mode synthesis |
| `reference_text` | Inline transcript of `voice_ref` |
| `reference_text_file` | Alternative to `reference_text`: load transcript from a `.txt` file |

Rules:

- `breeze-base` with `"lora": null` is required (unadapted base).
- Each adapter id must be unique; each `lora` path must be a directory that
  contains a valid Instavar package.
- If `voice_ref` is set, you must also set exactly one of `reference_text` or
  `reference_text_file`. Startup fails if the pair is incomplete or the files
  are missing.
- When a model has `voice_ref`, speech requests for that id automatically use
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
docker run --gpus all -p 8080:8080 \
  -v /path/to/models:/models:ro \
  -v /path/to/loras:/loras:ro \
  -v /path/to/server.json:/app/server.json:ro \
  local/breeze-lora-server:cuda12
```

Select the adapter with the OpenAI `model` field:

```bash
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{"model":"my-voice","input":"The train arrives in five minutes.","seed":42}'
```

Use `"model":"breeze-base"` for the unadapted path.

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/audio/speech` (complete WAV only)

Select adapters with the OpenAI `model` field. `breeze-base` is the unadapted base.

## Request logging

Each `/v1/audio/speech` job logs one line (request text is never logged). Logging
is enabled by default to stdout; pass `--log-file path` to append to a file.

Example:

```text
[info] [breeze_lora_server] speech request model=<id> first_load=<true|false> load_ms=<ms> generate_ms=<ms> status=ok
```

- `first_load=true` on the first live activation of that model id in the process
  (`breeze-base` is activated at startup, so its first request is not a first load).
- `load_ms` is the adapter/base switch time; switching only rebinds side-adapter
  buffers, so this is normally `0.0`, and it is `0.0` when the requested model is
  already active.
- Failures use `status=error` with the same timing fields (still no request text).

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
