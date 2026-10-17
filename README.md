# Strata on 2× Tesla V100 (Volta) — Docker + Dockge

Run [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (125B-total / 6B-active
hybrid MoE) with [Strata](https://github.com/Niko1221/Strata) on a pair of Volta cards, as a
Dockge-managed Docker stack.

**Measured on 2× V100-PCIE-32GB, Unsloth UD-IQ4_XS, int8 KV, MTP spec-4, 128K context:
53–58 tok/s decode — ~45% faster than the llama.cpp baseline (37.3 tok/s) on the same
weights and GPUs — with vision working and both cards layer-split to ~32 GB each.**

## Why a patched image is required

Strata's stock `Dockerfile` cannot run on Volta:

1. It builds on `nvidia/cuda:13.0.0-devel-ubuntu24.04`, and **CUDA 13 dropped Volta** (sm_70).
2. Strata's `CMakeLists.txt` refuses `CUDA_ARCHITECTURES` below 75 unless
   `-DSTRATA_EXPERIMENTAL_SM60=ON` is set — the flag that swaps in the Volta-safe kernels
   (BF16 projections on FP16 tensor cores, `mma.m8n8k4` prompt attention, dp4a expert kernels;
   see Strata's `docs/NVIDIA_V100.md` and `docs/OLDER_GPUS.md`).

`Dockerfile.volta.patch` in this repo makes the three changes: CUDA 12.9 base image, the
experimental flag on the engine build, and on the vision-encoder build.

## Build

```bash
git clone https://github.com/Niko1221/Strata
cd Strata
git apply /path/to/Dockerfile.volta.patch   # or the sed one-liners below

docker build --build-arg CUDA_ARCHITECTURES=70 -t strata-v100:local .
```

Equivalent sed, if you prefer:

```bash
sed -i \
  -e 's|FROM nvidia/cuda:13.0.0-devel-ubuntu24.04|FROM nvidia/cuda:12.9.1-devel-ubuntu24.04|' \
  -e 's|"-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_Q6K_EXPERTS=ON"|"-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_EXPERIMENTAL_SM60=ON", "-DSTRATA_Q6K_EXPERTS=ON"|' \
  -e 's|"-DSTRATA_VISION_CUDA=ON", "-DSTRATA_PORTABLE=OFF"|"-DSTRATA_VISION_CUDA=ON", "-DSTRATA_EXPERIMENTAL_SM60=ON", "-DSTRATA_PORTABLE=OFF"|' \
  Dockerfile
```

Host requirements: NVIDIA driver 580+, nvidia-container-toolkit, ~95 GB of GGUFs (see below).
A CUDA toolkit on the host is **not** needed — everything compiles inside the image.

### First start: bake the Volta engine back into the image

On the first container start, Strata's setup detects Volta and compiles its own
`engine-cuda12` inside the container (~9 min on 64 cores). That layer is ephemeral — a
container recreate would redo it. Bake it into the image once the server is up:

```bash
docker commit <container> strata-v100:local
```

After that, a full `docker compose down && up` reaches healthy in ~2–3 minutes
(model load only: no rebuild, no re-verification).

## Weights

The stack reuses Unsloth UD-IQ4_XS shards in place (no download, no copy; SHA-256 verified):

```
models/Qwen3.8-Flash-Next/
├── Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf   ( 11 MB)
├── Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf   (47 GB)
├── Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf   (41 GB)
└── mmproj-F16.gguf                                     (863 MB)
```

from [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).
The first start still downloads the vision encoder (~0.9 GB) and the MTP draft layer (~6 GB)
into the named volume. Without pre-downloaded shards, drop `GGUF_DIR` and let setup fetch
everything (~94 GB).

## The stack

`compose.yaml` (adjust the models path and port to taste):

```yaml
services:
  strata:
    image: strata-v100:local
    container_name: qwen38-flash-next-strata
    restart: unless-stopped
    ports:
      - "8184:8184"
    environment:
      - FAMILY=unsloth
      - MODEL=UD-IQ4_XS
      - CONTEXT=131072
      - VISION=yes
      - KV=int8
      - GPUS=0,1
      - PORT=8184
      - HOST=0.0.0.0
      - GGUF_DIR=/models/Qwen3.8-Flash-Next
    volumes:
      - /home/YOURUSER/models:/models:ro
      - strata-data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ulimits:
      memlock: -1
    stop_grace_period: 120s
volumes:
  strata-data:
```

Drop the directory into your Dockge stacks folder and deploy. Notes:

- `GGUF_DIR` = the mounted shards; `strata-data` holds only the pack, MTP layer and install config.
- `KV=int8` + `GPUS=0,1` match Strata's documented V100 reference configuration.
- `stop_grace_period` matters: Strata releases page-locked RAM slowly.
- To change model settings later, add `REINSTALL=1` to `environment` once and recreate.
- `/models` is read-only on purpose; Strata logs a harmless "finish mark cannot be written"
  during the initial setup pass because of it.
- OpenAI-compatible API at `http://<gpu-host>:8184/v1` (plus an Anthropic-style
  `/v1/messages`), tool calling works, web chat UI at `/`. Add `API_KEY` before exposing
  beyond a trusted LAN.

## Point a client at it (Oh My Pi example)

`examples/omp-strata.yml` shows an [Oh My Pi](https://github.com/can1357/oh-my-pi) provider
entry; anything OpenAI-compatible (continue, llama-cpp clients, OpenWebUI, …) works the same way:

```yaml
providers:
  strata:
    baseUrl: http://<gpu-host>:8184/v1
    api: openai-completions
    auth: none
    models:
    - id: qwen3.8-flash-next-unsloth-ud-iq4_xs
      name: Qwen3.8-Flash-Next UD-IQ4_XS (Strata, 2x V100)
      reasoning: true
      input: [text, image]
      contextWindow: 131072
      maxTokens: 16384
      supportsTools: true
```

## Benchmarks (2× V100-PCIE-32GB, driver 580)

| Metric | Strata 2× V100 | llama.cpp 2× V100 (same weights) | Single V100 (Strata docs) |
|---|---|---|---|
| Decode, 256 tok, temp 0.7 | **53.1–57.5 tok/s** | 37.3 tok/s | 38–51 tok/s |
| Prompt processing | ~650 tok/s (round-trip) | — | 1,100–1,250 tok/s (28K+ prompts) |

VRAM: ~32 GB on each card (a small TTS container coexisted on GPU 1). RAM budget for
experts: 23 GiB pinned; with 64 GB total VRAM all ~59.5 GB of routed experts stay resident
— no SSD spill.

## Server hardware (the benchmark machine)

| Component | Detail |
|---|---|
| GPU | 2× Tesla V100-PCIE-32GB (sm_70), PCIe Gen3 x16 each, no NVLink — layer split needs only host RAM hand-off |
| CPU | 2× Intel Xeon E5-2682 v4 @ 2.50 GHz (Broadwell-EP, 2 sockets, 2 NUMA nodes, 64 threads) |
| RAM | 47 GiB (Strata pins ~23 GiB for the expert cache; the rest of the experts live in VRAM) |
| Disk | 500 GB virtual disk (this runs in a QEMU/Q35 VM with both V100s passed through — no bare metal required) |
| OS / driver | Debian 13 (trixie), kernel 6.12, NVIDIA driver 580.173.02, nvidia-container-toolkit |

Nothing exotic: a 2016-era dual-socket board with a pair of secondhand V100s and 48 GB of RAM
is enough for the numbers above. AVX2 CPUs are required by the prebuilt engine; without AVX2,
build the image on the target machine itself.

## Credits

- [Strata](https://github.com/Niko1221/Strata) — the engine and quant tooling; read
  `docs/NVIDIA_V100.md`, `docs/OLDER_GPUS.md`, `docs/MULTI_GPU.md`, `docs/UNSLOTH_Q4.md`.
- [Unsloth](https://huggingface.co/unsloth) — the UD-IQ4_XS quantization.
- [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — the model.

Configuration files here are provided as-is (MIT-style: do what you want; no warranty).
