# breeze-lora-server

OpenAI-compatible HTTP server for **Breeze TTS 2** with **LoRA adapter hot-swap**.

One resident Breeze GGUF base model serves many named voices. The OpenAI model
identity is always `breeze-base`; each Instavar-format adapter or
instruction/reference preset is a selectable `voice`, and the server rebinds a
per-adapter low-rank side path when the requested voice changes — the resident
base weights are never rewritten. Training and adapter export stay in Python;
this repository only **serves** adapters.

This project is a focused fork of [`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp)
that keeps the upstream GGUF runtime, CUDA/CPU backends and HTTP stack, and adds a
dedicated Breeze server executable with a LoRA registry.

| | |
|---|---|
| Upstream revision | [`ed96b7307c8daba2ebcf7912af928825f6b14cb9`](https://github.com/0xShug0/audio.cpp/commit/ed96b7307c8daba2ebcf7912af928825f6b14cb9) (see `UPSTREAM_PIN.txt`) |
| Server source | [`app/breeze_lora_server/`](app/breeze_lora_server) |
| LoRA runtime | [`src/models/breeze_tts/lora.cpp`](src/models/breeze_tts/lora.cpp), [`include/engine/models/breeze_tts/lora.h`](include/engine/models/breeze_tts/lora.h) |
| License | Apache-2.0 for the code; the Breeze TTS 2 **weights** have their own license (see below) |

---

## Capability status

Read this before deploying.

| Capability | Status |
|---|---|
| `GET /health` | Implemented |
| `GET /v1/models` | Implemented — singleton `breeze-base` only |
| `POST /v1/audio/speech` | Implemented — complete `audio/wav` (default) or `audio/mpeg` (`response_format=mp3`); `model` must be `breeze-base`, optional `voice` |
| Bounded FIFO queue with `503 queue_full` | Implemented — one inference worker, `max_queue_depth` bound |
| Voice selection by request `voice` id | Implemented — re-selecting the active adapter/base is a no-op |
| Resident base with a **non-destructive side-adapter** path | Implemented — see the side-adapter note below |

### Resident base and the side-adapter path

The base GGUF stays resident and immutable. Each adapter is stored as its own
low-rank `A`/`B` pair in the Breeze weight store, and switching adapters changes
only which pair the per-forward graph binds:

- Adapted projections compute `y = W·x + scale·B·(A·x)` as extra graph nodes;
  the resident base tensors are never rewritten.
- `BreezeLoraManager::activate_adapter()` / `activate_base()` only change the
  active id and side-branch selection. They do not merge weights or re-upload
  dense tensors, so switch cost does not scale with adapter-covered weight bytes.
- Base rows are captured once at bind time (`upload_base_live_buffers()`) and
  `base_rows()` exposes them for inspection; `activate_*` does not touch them.

Coverage for this invariant:

- `tests/unittests/test_breeze_lora_side_adapter.cpp` — real ggml graph
  proof that the enabled path equals `base(x) + scale·B(A(x))` and the disabled
  path equals `base(x)` exactly.
- `tests/unittests/test_breeze_lora_activation.cpp` — drives
  `breeze-base → adapter-a → adapter-b → adapter-a → breeze-base` through a real
  bound backend tensor and asserts the resident base bytes are unchanged after
  every switch, with reselect as a no-op.
- `breeze_lora_linear_forward()` (host-side reference implementation of the same
  math) remains unit-tested in `tests/unittests/test_breeze_lora_math.cpp`.
- `tests/unittests/test_decoder_packed_hook.cpp` — builds a `PackedGateUp` decoder
  layer with the side-adapter hook installed and asserts the hook path produces the
  same result as the packed path. This covers the interaction between the hook and
  the packed-gate/up MLP layout: the hook splits the packed tensor, so the
  single-tensor fused swiglu kernel cannot be used and the split path is taken.
  Before that split was handled, a CPU speech request crashed with SIGSEGV during
  decoder graph build (`ggml_glu_impl` on a null tensor); CUDA was unaffected.

The disabled/base path is exact: a null adapter context returns the plain
`LinearModule` result, so `breeze-base` output is byte-identical to an
unadapted projection.

---

## Quick start (Docker, CPU)

The fastest way to see it work. CPU generation is slow; use CUDA for real traffic.

```bash
docker build -f .devops/breeze_lora_cpu.Dockerfile -t breeze-lora-server:cpu .
```

Create a model directory containing `server.json` plus everything it references:

```
my-models/
├── server.json
├── Breeze-TTS-2-GGUF/          # Breeze TTS 2 GGUF package
└── loras/
    └── my-voice/
        ├── adapter_config.json
        ├── adapter.safetensors
        ├── reference.wav
        └── reference.txt
```

Run it:

```bash
docker run --rm -p 8080:8080 \
  -v /path/to/my-models:/models:ro \
  breeze-lora-server:cpu
```

The image entrypoint already passes `--config /app/server.json`; mount your config to
that path, or override the entrypoint. With the bundled compose file:

```bash
BREEZE_MODELS_DIR=/path/to/my-models \
  docker compose -f .devops/docker-compose.breeze-models.yml up -d
```

### CUDA

```bash
docker build -f .devops/breeze_lora_cuda.Dockerfile \
  --build-arg CUDA_DOCKER_ARCH="80;86;89;90" \
  -t breeze-lora-server:cuda .

docker run --rm --gpus all -p 8080:8080 \
  -v /path/to/my-models:/models:ro \
  -v /path/to/server.json:/app/server.json:ro \
  breeze-lora-server:cuda
```

Prefer mounting the config inside `/models` (and passing `--config /models/server.json`) so relative `lora` / `voice_ref` / `base_model` paths resolve inside the same tree. The Unraid templates below do that.

### Unraid

Pre-built images are on GHCR:

| Template | Image |
|---|---|
| CUDA | `ghcr.io/doguitar/breeze-tts-lora-server:cuda` |
| CPU | `ghcr.io/doguitar/breeze-tts-lora-server:cpu` |

XML templates live in [`.devops/unraid/`](.devops/unraid/).

#### Install the template from the Unraid CLI

On the Unraid server (SSH or web terminal), download a template into dockerMan's user-templates directory, then add the container from the Docker UI:

```bash
# CUDA (recommended)
mkdir -p /boot/config/plugins/dockerMan/templates-user
curl -fsSL -o /boot/config/plugins/dockerMan/templates-user/my-breeze-lora-server-cuda.xml \
  https://raw.githubusercontent.com/doguitar/breeze-tts-lora-server/main/.devops/unraid/breeze-lora-server-cuda.xml

# Optional: CPU template
curl -fsSL -o /boot/config/plugins/dockerMan/templates-user/my-breeze-lora-server-cpu.xml \
  https://raw.githubusercontent.com/doguitar/breeze-tts-lora-server/main/.devops/unraid/breeze-lora-server-cpu.xml
```

Then in the Unraid web UI:

1. Put your model tree on the array (default template path: `/mnt/user/appdata/breeze-models`) with `server.json`, `Breeze-TTS-2-GGUF/`, and any `loras/` adapters. For CUDA set `"backend": "cuda"`; for CPU use `"backend": "cpu"`.
2. **CUDA only:** install the **Nvidia Driver** plugin, reboot if prompted, and note the GPU UUID under **Plugins → Nvidia Driver**.
3. **Docker → Add Container**. Open the template dropdown and select **breeze-lora-server-cuda** (or **-cpu**). Unraid prefixes user templates with `my-` on disk; the template `<Name>` is what appears in the UI.
4. Set **Models** to your host path if it differs from the default. For CUDA, set **NVIDIA_VISIBLE_DEVICES** to your GPU UUID (or leave `all`).
5. Apply / Start. Check `http://<unraid-ip>:8080/health`.

The templates set `PostArgs` to `--config /models/server.json` (last `--config` wins over the image entrypoint) and, for CUDA, `ExtraParams` to `--runtime=nvidia --restart=unless-stopped` plus the usual NVIDIA env vars.

#### Or run directly with `docker` on Unraid

```bash
# CUDA — replace GPU UUID from Plugins → Nvidia Driver when pinning a device
docker run -d --name breeze-lora-server-cuda --net bridge \
  --runtime=nvidia --restart=unless-stopped \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -p 8080:8080 \
  -v /mnt/user/appdata/breeze-models:/models:ro \
  ghcr.io/doguitar/breeze-tts-lora-server:cuda \
  --config /models/server.json

# CPU
docker run -d --name breeze-lora-server-cpu --net bridge \
  --restart=unless-stopped \
  -p 8080:8080 \
  -v /mnt/user/appdata/breeze-models:/models:ro \
  ghcr.io/doguitar/breeze-tts-lora-server:cpu \
  --config /models/server.json
```

---

## Configuration

`server.json` is resolved relative to its own directory, so relative paths inside it
stay inside the mounted tree.

```json
{
  "host": "0.0.0.0",
  "management": true,
  "port": 8080,
  "backend": "cuda",
  "device": 0,
  "threads": 1,
  "max_queue_depth": 8,
  "base_model": "Breeze-TTS-2-GGUF",
  "base_revision": "799624c0b4a1daa8db6d28bbd9850043c0270734",
  "voices": [
    {
      "id": "my-voice",
      "lora": "loras/my-voice",
      "default_instruction": "Speak clearly and naturally.",
      "voice_ref": "loras/my-voice/reference.wav",
      "reference_text_file": "loras/my-voice/reference.txt"
    },
    {
      "id": "base-narrator",
      "lora": null,
      "default_instruction": "Speak clearly and naturally."
    }
  ]
}
```

| Field | Meaning |
|---|---|
| `host`, `port` | Listen address. Port must be 1–65535. |
| `management` | Optional boolean, default `false`. `true` allows preset writes (Save preset, Clear reference, New preset id) even when `host` is not loopback, including a published Docker port on `0.0.0.0`. Anyone who can open the page can then write presets. Absent or `false` keeps preset writes limited to a loopback bind (`127.0.0.1`, `localhost`, or `::1`). A non-boolean value is a config error. Generate / audition does not depend on this flag. |
| `backend` | `cuda` or `cpu`. |
| `device` | Backend device index. |
| `threads` | Worker threads for the backend. Must be positive. |
| `max_queue_depth` | Max waiting requests before `503 queue_full`. Must be positive. |
| `base_model` | Breeze TTS 2 GGUF package directory or file. Must exist. |
| `base_revision` | Optional; checked against each adapter manifest's pinned base revision. |
| `voices[].id` | Voice id selected by the speech `voice` field. Must be unique. |
| `voices[].lora` | Adapter directory, or `null`/omitted for an instruction/reference-only base voice. |
| `voices[].default_instruction` | Instruction used when a request omits `instruction`. |
| `voices[].voice_ref` | Optional reference WAV that puts the voice into clone mode. |
| `voices[].reference_text` / `reference_text_file` | Transcript of `voice_ref`. Exactly one. |

Rules enforced at startup (the process refuses to start rather than skipping a bad entry):

- `voices` must be a JSON array (empty is allowed for base-only). Legacy `models` is rejected.
- Voice ids must be unique; non-null adapter directories must exist.
- `voice_ref` requires exactly one of `reference_text` / `reference_text_file`.
- Each adapter must validate: schema, variant, rank, alpha, checksums, base identity,
  and a complete A/B tensor pair for every target module.

An invalid deployment is a startup failure by design.

### Adapter format

Each adapter directory contains `adapter_config.json` and `adapter.safetensors`.
The server accepts exactly one layout in v1:

| Field | Required value |
|---|---|
| `schema_version` | `1` |
| `artifact_type` | `breeze_lora_adapter` |
| `lora.variant` | `backbone_depth_projection` |
| `lora.rank` | `8` |
| `lora.alpha` | positive finite (scale = `alpha / rank`) |

Adapted modules: backbone and depth-decoder attention/MLP projections
(`.self_attn.{q,k,v,o}_proj`, `.mlp.{gate,up,down}_proj`), plus `text_encoder_proj`,
`depth_decoder.model.inputs_embeds_projector`, and `lm_head`. Tensors are named
`{module}.lora_A` / `{module}.lora_B` and map to base weight `{module}.weight`.

Any other variant or rank is rejected at startup rather than adapted dynamically.
See [`app/breeze_lora_server/README.md`](app/breeze_lora_server/README.md) for the full
manifest schema and a worked `adapter_config.json`.

---

## API

### `GET /health`

```json
{"status":"ok"}
```

### `GET /v1/models`

```json
{"object":"list","data":[
  {"id":"breeze-base","object":"model","owned_by":"breeze-lora-server"}
]}
```

### `POST /v1/audio/speech`

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

Omit `voice` for the unadapted base:

```bash
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o out.wav \
  -d '{"model":"breeze-base","input":"The train arrives in five minutes.","seed":42}'
```

| Field | Meaning |
|---|---|
| `model` | Required. Must be `breeze-base`. Any other value returns `400`. |
| `voice` | Optional configured voice id. Absent => unadapted base. Unknown ids return `400`. |
| `input` | Required, non-empty text. |
| `instruction` | Optional voice direction. Falls back to the voice's `default_instruction`, or the built-in instruction when `voice` is omitted. |
| `voice_ref` / `reference_text` | Optional request-level clone override. Must be supplied **together**. |
| `response_format` | Optional. Non-streaming formats: `wav` (default, `audio/wav`) or `mp3` (`audio/mpeg`). |
| `seed`, `guidance_scale`, `temperature`, `depth_temperature`, `top_k`, `top_p`, `max_tokens` | Breeze sampling controls. |

Returns a complete audio body: `audio/wav` (RIFF/WAVE) by default, or `audio/mpeg`
when `response_format` is `mp3`. `stream` and `stream_format` are rejected with
`400`. Any `response_format` other than `wav` or `mp3` is also rejected with `400`.

Errors are JSON with a stable `type`:

| Status | `type` | When |
|---|---|---|
| `400` | `invalid_request_error` | Missing/invalid fields, unknown model/voice, unsupported format |
| `503` | `server_error` | `queue_full` — more than `max_queue_depth` requests waiting |
| `500` | `server_error` | Synthesis failure |

#### Request logging

Successful speech jobs log one timing line. Request text is not included on success.

```text
[info][breeze_lora_server] speech request model=breeze-base voice=<id|(none)> first_load=<true|false> load_ms=<ms> generate_ms=<ms> status=ok
```

- `first_load=true` on the first live activation of that activation id in the process.
  `breeze-base` is activated at startup, so its first request is not a first load.
- `load_ms` is the adapter/base switch time; switching only rebinds side-adapter
  buffers, so it is `0.0` in practice and when the voice is already active.
- Synthesis failures also log the same fields with `status=error`.
- Any HTTP response other than `200` additionally logs full request details
  (method, path, query, headers, body) plus the response status and body.

Logging is on by default to stdout; `--log-file <path>` appends to a file instead.

---

## Building and testing

### Container verification

```bash
.devops/verify_docker.sh cpu    # build CPU image, then run unit tests inside it
.devops/verify_docker.sh cuda   # build CUDA image only (tests need a GPU host)
```

`DOCKER_EXIT=0` in the log/marker means the build (and, for CPU, the unit tests)
succeeded. On Windows, `.devops/verify_cpu_docker.ps1` is the equivalent harness.

These scripts — and the CI workflows — build and run exactly the three binaries
listed in `verify_docker.sh`: `breeze_lora_math_test`, `breeze_lora_manifest_test`,
`breeze_lora_server_config_test`. The activation, side-adapter and packed-hook tests
are **not** in that set; the native command below builds and runs all six, and needs
`-DENGINE_BUILD_MODEL_TESTS=ON` for the packed-hook test.

The CUDA compile outlives a single shell invocation, so
`.devops/build_cuda_detached.sh <tag>` launches it with `nohup` and writes
`.build/cuda-<tag>.log` plus a `.build/cuda-<tag>.exit` marker holding
`DOCKER_EXIT`. `.build/` is gitignored local build state.

### Unit tests

| Binary | Coverage |
|---|---|
| `breeze_lora_math_test` | Side-path math `W·x + scale·B·(A·x)`, disabled path equals `W·x`, merge equivalence |
| `breeze_lora_manifest_test` | Manifest schema/variant/rank validation and adapter checksums |
| `breeze_lora_activation_test` | `breeze-base → adapter-a → adapter-b → adapter-a → breeze-base` activation state, no-op reselect, resident base rows unchanged |
| `breeze_lora_side_adapter_test` | Real ggml graph: enabled path equals `base(x) + scale·B(A(x))`, disabled path equals `base(x)` |
| `breeze_lora_server_config_test` | Config parsing, path resolution, required-pair rules, duplicate id rejection |
| `decoder_packed_hook_test` | `PackedGateUp` MLP with the side-adapter hook installed builds and matches the packed path |

The `decoder_packed_hook_test` (like the pre-existing `decoder_packed_projection_test`)
is registered under `ENGINE_BUILD_MODEL_TESTS`, so configure with
`-DENGINE_BUILD_MODEL_TESTS=ON` to build it.

### Native build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_ENABLE_CUDA=OFF -DENGINE_BUILD_TESTS=ON -DENGINE_BUILD_MODEL_TESTS=ON
cmake --build build --parallel --target breeze_lora_server \
  breeze_lora_math_test breeze_lora_manifest_test breeze_lora_activation_test \
  breeze_lora_side_adapter_test breeze_lora_server_config_test decoder_packed_hook_test

# Run them (each prints "<name>: ok" on success).
for t in breeze_lora_math_test breeze_lora_manifest_test breeze_lora_activation_test \
         breeze_lora_side_adapter_test breeze_lora_server_config_test decoder_packed_hook_test; do
  ./build/bin/$t || exit 1
done
```

### Smoke test

With a server running:

```bash
SMOKE_VOICES="adapter-a adapter-b" \
  ./scripts/smoke_breeze_lora_server.sh
```

It checks `/health`, `/v1/models` (singleton `breeze-base`), requests a no-voice base
WAV plus each configured voice with fixed `model=breeze-base`, verifies every default
response is RIFF/WAVE, requests an explicit `response_format=mp3` response and checks
`Content-Type: audio/mpeg` plus `ffprobe` `format_name=mp3`, and confirms
`model=adapter-a` and `voice=missing` each return `400`. Requires `curl`, `python3`,
and `ffprobe` (from FFmpeg).

---

## Continuous integration

| Workflow | Purpose |
|---|---|
| [`.github/workflows/cpu-image.yml`](.github/workflows/cpu-image.yml) | Builds the CPU image, runs the three unit-test binaries inside it, pushes `:cpu` and `:cpu-<sha>` to GHCR |
| [`.github/workflows/cuda-image.yml`](.github/workflows/cuda-image.yml) | Builds the CUDA image with a configurable `CUDA_DOCKER_ARCH`, pushes `:cuda` and `:cuda-<sha>` to GHCR |

Both workflows are path-filtered to the Breeze server surface and can be run manually
via `workflow_dispatch`. The CUDA architecture list defaults to `80;86;89;90` and is
passed to CMake as `-DCMAKE_CUDA_ARCHITECTURES`; each entry is a bare SM number
(`80`) or a suffixed form (`86-real`, `90-virtual`). Keeping this list explicit
matters: without it the engine falls back to its portable default architecture list
and compiles every `.cu` file once per architecture.

---

## Side-adapter binding

The side path is wired through the decoder build hooks rather than through
`LinearModule` itself:

1. `DecoderStackConfig::side_adapter_layer_linear` and
   `DecoderLayerConfig::side_adapter_linear` (`include/engine/framework/modules/transformers/decoder.h`)
   let the owning model override how a projection is built.
2. `build_adapter_linear()` (`src/models/breeze_tts/lora_linear.cpp`) emits
   `base(x)` and, when an adapter is active,
   `ggml_mul_mat(a, x)` → `ggml_mul_mat(b, ·)` → `ggml_scale` → `ggml_add`.
3. `BreezeGeneratorRuntime` installs those hooks for the backbone and depth
   stacks and resolves the active adapter's `A`/`B` per module name at graph build
   time, so switching an adapter changes only the bound buffers.

Packed targets (`qkv_weight`, `gate_up_proj`) are sliced per projection before the
hook runs, so each projection gets its own side path. Because that split replaces
the single packed tensor, the packed-gate/up fused swiglu kernel is not used on the
hook path; the split swiglu path is used instead, which produces the same result.

---

## Licensing

| Artifact | License |
|---|---|
| This repository's code (incl. `app/breeze_lora_server`, `src/models/breeze_tts/lora.cpp`) | Apache-2.0, inherited from upstream `audio.cpp` (© ShugoAI LLC) |
| Breeze TTS 2 **weights** and any adapter / fine-tune / self-hosted derivative | [BreezeBlue Research and Non-Commercial License](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE) |

LoRAs and fine-tunes are **Derivative Models** under the BreezeBlue agreement:
research and non-commercial use only unless you hold a separate commercial license
from BreezeBlue / RESONIA. Commercial use of the weights is **not** granted by this
repository's Apache-2.0 code license. Training tooling is third-party and independent
of BreezeBlue; see [`app/breeze_lora_server/README.md`](app/breeze_lora_server/README.md).

## Provenance

Upstream project: [`0xShug0/audio.cpp`](https://github.com/0xShug0/audio.cpp) at
`ed96b7307c8daba2ebcf7912af928825f6b14cb9`. Upstream model packages live in the
`audio-cpp` HuggingFace repositories; the upstream README and `docs/` are retained for
the wider model catalog this runtime can load.
