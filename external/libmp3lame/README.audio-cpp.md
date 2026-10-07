# audio.cpp vendoring notes

This directory vendors the **libmp3lame 3.100** encoder sources from the upstream
LAME release (`lame-3.100`).

Upstream license files kept here:

- `COPYING` — GNU LGPL v2
- `LICENSE` — commercial-use / linking notes from the LAME project
- `README` — upstream project readme

Only the encoder library sources under `libmp3lame/` and the public header
`include/lame.h` are included. The mpglib decoder is disabled via `config.h`
(`HAVE_MPGLIB` undefined). No host-installed LAME, ffmpeg, or shell-out encoder
is required; the sources compile into the static `audiocpp_libmp3lame` target
and are linked only by `breeze_lora_server`.
