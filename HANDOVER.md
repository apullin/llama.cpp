# Tenstorrent Metalium backend handover

Updated: 2026-10-02

## Key finding (2026-10-02): the card is fast; this backend is not

Tenstorrent's own stack runs Qwen3-8B on the N300 at 34.9 tok/s at batch 1 and 873.9 tok/s aggregate at batch 32. On the same card, this llama.cpp backend gets 7.2 and 39.2 tok/s. The tuned host CPU gets 20.7 and 160.2 tok/s.

- **Software, not silicon.** The gap is 4.9x at batch 1 and 22x at batch 32.
- **Both ASICs hold layers here, but they never overlap.** Splitting a step into more ubatches costs proportionally more.
- **The second ASIC adds capacity, not speed.**
- **The llama-server path collapses under concurrency and crashes on prompt-state restore.**
- Evidence: `evidence/20261002-glimmer-v079/README.md`, "Batch scaling, both ASICs, and Tenstorrent's own stack".

Decision point before more backend work: either move N300 serving to Tenstorrent's stack (tt_transformers / TT-vLLM) for the models it supports, or target the backend's structural limits first. Those limits are host-side per-node dispatch on one thread, no trace replay for batched shapes, and no cross-ASIC overlap.

Running TT's stack:

```bash
T=/home/pullin/personal/tenstorrent/generated/tt-metal-v0.79.0
cd $T && TT_METAL_HOME=$T PYTHONPATH=$T/ttnn:$T ARCH_NAME=wormhole_b0 HF_MODEL=Qwen/Qwen3-8B MESH_DEVICE=N300 \
  TT_CACHE_PATH=$HOME/.cache/tt_transformers/Qwen/Qwen3-8B/N300 python_env/bin/python -m pytest -s \
  "models/tt_transformers/demo/simple_text_demo.py::test_demo_text[wormhole_b0-mesh_device0-device_params0-False-performance-ci-32]"
```

- `python_env` is a uv venv. It is gitignored in the tt-metal tree.
- Glimmer has no tt_transformers implementation.
- TT's Qwen3.5/3.6 code is Blackhole-only.

## Objective

Continue the private `tt-dual-asic` llama.cpp Metalium backend for one Tenstorrent N300 (two Wormhole ASICs) on **TT-Metal v0.79.0, the only supported runtime**:

1. Keep the dual-ASIC implementation working.
2. Beat the tuned host CPU on Muse Glimmer.
   - Trace capture is back on v0.79: 7.01 tok/s decode, and 7.18 with the attention-gate overlap.
   - The CPU does 10.67 tok/s (`evidence/20261002-glimmer-v079/`).
3. Test and commit each pending optimization independently.
4. Produce LocalMaxxing evidence. Submit N300 runs only after LocalMaxxing registers an N300 hardware identity.

## Runtime and repository

- **llama.cpp:** `/home/pullin/personal/serving/llama-upstream`, branch `tt-dual-asic`, HEAD `67922c248`. It is rebased onto upstream `4da6337767f9`. Recent commits:
  - `ebfc9aa11`: v0.79 API port
  - `4376b5c03`: close submeshes before the parent at exit, so the fabric routers terminate
  - `67922c248`: remove the old-runtime compatibility layer
- **TT-Metal:** `/home/pullin/personal/tenstorrent/generated/tt-metal-v0.79.0` (`de546d3b14`). It is a worktree of `/home/pullin/personal/tenstorrent/repos/tt-metal` and carries one local patch (below). It is checked out on branch `qwen36-n300-mtp`: `glimmer-port` (Glimmer port, decode logits readback) plus the qwen36 speculative decoding commits.
- **TT-Metal full rebuild** (needed after the 2026-10-03 cleanup deleted `build_Release`): `PATH=/home/pullin/personal/tenstorrent/generated/toolbin:$PATH ./build_metal.sh --enable-ccache` from the tree. `toolbin` maps the pinned `clang-20`/`clang++-20` names to the system clang 22 and provides CMake 3.31 (`generated/cmake-venv`). The system CMake 3.28 breaks the googletest build: without policy CMP0180, CPM's `gtest_SOURCE_DIR` shadows googletest's own and its include paths point at the wrong directory. The llama.cpp Metalium build is then `TT_METAL_HOME=<tree> cmake -B build-tt079 -DCMAKE_BUILD_TYPE=Release -DGGML_METALIUM=ON -DLLAMA_CURL=OFF`.
- **Build directories:**
  - `llama-upstream/build-tt079`: the Metalium build. Rebuild with `TT_METAL_HOME=<tt-metal tree> cmake --build build-tt079 -j 64`.
  - `llama-upstream/build-rebase-check`: CPU build of the same tree. It is the correctness reference, the CPU control for the `cpu-*` `bench.py` targets, and the binary for the `cpu` modes of `run-dflash.sh` and `run-dspark.sh`. It has `llama-server`, `llama-bench`, `llama-batched-bench`, `llama-perplexity`, `llama-quantize`, `llama-completion`, and `llama-speculative-simple`. Tuned CPU flags are in `KNOWLEDGE.md`, "Host CPU baseline".
  - `llama-upstream/build-cuda`: production CUDA server, built from commit `fe57b804e` in the detached worktree `llama-upstream/build-cuda-src` (`cmake -S build-cuda-src -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`). Don't rebuild it casually.
- **Run environment** (also in `localmaxxing-n300/runtimes.toml`):

```bash
TT79=/home/pullin/personal/tenstorrent/generated/tt-metal-v0.79.0
export TT_METAL_RUNTIME_ROOT="$TT79"
export LD_LIBRARY_PATH="$TT79/build/tt_metal:$TT79/build/ttnn:$TT79/build/lib:$TT79/build/tt_stl:$TT79/build/tt_metal/third_party/umd/lib"
```

  `build/ttnn` must come before `build/lib`, because only `build/ttnn` has the patched `_ttnncpp.so`.

- **Staged-commit validation worktree:** `/tmp/llama-tt079-compat-validate`, at `67922c248`, with its own `build79`. To validate a commit without the WIP, edit and build there, then commit there and move `refs/heads/tt-dual-asic` to the new commit.
- **Old runtime:** TT-Metal v0.74 (`13adda80`) and everything built on it were deleted on 2026-10-02. Don't reintroduce it. Its local changes are kept as branches `uint32-remainder-support` and `paged-update-cache-compact-cb` in `repos/tt-metal`. Older `NOTES.md` entries and `evidence/` refer to it.

## Safety rules

- Never use `pkill -f llama-server` or any broad process kill. Stop only exact benchmark PIDs.
- The CUDA `llama-server` is production infrastructure. Don't stop or restart it.
- Run one TT workload at a time, and check `fuser /dev/tenstorrent/0` first.
- Resets: `~/.tenstorrent-venv/bin/tt-smi -r` is approved whenever it is needed.
  - Clean exits no longer leave the card wedged (`4376b5c03`).
  - A crash, kill, or timeout skips fabric teardown, and the next start then fails with "Timed out waiting for ETH heartbeat". That case needs one reset.
  - Details are under "Fabric router teardown" in `KNOWLEDGE.md`.
- Never label N300 results as P150.
- Commit only after a clean build and a deterministic device validation, and keep each commit separate from the WIP.

## Local TT-Metal patch

The v0.79 tree has one uncommitted change, in `ttnn/cpp/ttnn/operations/experimental/paged_cache/device/update_cache/paged_update_cache_program_factory.cpp`. The same patch is preserved as branch `paged-update-cache-compact-cb`.

```cpp
const bool compact_output_cb = std::getenv("TTNN_PAGED_UPDATE_CACHE_COMPACT_OUTPUT_CB") != nullptr;
uint32_t num_output_tiles = compact_output_cb ? Wt : B * Wt;
```

- **Why it is needed:** stock v0.79 allocates `B * Wt` output tiles. The Qwen DFlash batched cache update then hits an L1 collision before inference: static dataflow buffers end at `1486048`, while an L1 buffer starts at `1367936`.
- **Why `Wt` is enough:** the writer consumes `Wt` tiles per head.
- **Rebuild:** `cmake --build <tree>/build --target _ttnncpp.so -j 64`.

## Current uncommitted llama.cpp work

Don't discard the dirty tree. It holds several independent experiments:

- **Direct tiled KV writes** (`direct_kv_write.{cpp,hpp}`, `kernels/direct_kv_write_dm.cpp`). Gate: `GGML_METALIUM_KV_DIRECT_WRITE=1`.
- **Zero-copy paged attention** (`paged_q_align.{cpp,hpp}`, `kernels/paged_q_align_dm.cpp`). Gate: `GGML_METALIUM_KV_PAGED_ATTENTION=1`.
- **Output-copy elision.** Gate: `GGML_METALIUM_ELIDE_OUTPUT_COPIES=1`. Optional counters: `GGML_METALIUM_OUTPUT_STORE_STATS=1`.
- **Fused cross-ASIC FFN SwiGLU.** Gate: `GGML_METALIUM_INTERNAL_FFN_SWIGLU=1`.
- **Attention-gate overlap** and related compiler changes.
- **Speculative-tree and DFlash changes** under `common/`, `examples/`, and `tools/completion/`.
- **Documentation** for this WIP in the dirty parts of `NOTES.md` and `KNOWLEDGE.md`.

Run `git status --short` before staging anything. Stage each feature on its own; don't mix experiments.

## DFlash baseline (v0.79 control)

`run-dflash.sh tt` runs on v0.79 (`build-tt079`, the environment above, and the DFlash gate set including `TTNN_PAGED_UPDATE_CACHE_COMPACT_OUTPUT_CB=1`):

```bash
QWEN_DFLASH_N_PREDICT=128 ./run-dflash.sh tt   # detached; log /tmp/qwen3-dflash-tt.log, PID in /tmp/qwen3-dflash-tt.pid
python3 /home/pullin/personal/serving/validate_qwen_dflash.py /tmp/qwen3-dflash-tt.log --require-metalium-gates
```

Warm control from 2026-10-02, on the quicksort code prompt (`evidence/20261002-v079-only/`):
- The validator passed with an exact greedy prefix.
- 128 tokens in 2.229 s, which is 57.424 tok/s.
- 112 of 219 drafts accepted (51.1%), over 16 target calls and 235 verifier rows.
- Fabric 1,366 direct, 0 fallback. Batched QKV 684/684. JIT 305/305.

This is the same-runtime control for every gate test below. The earlier best of 54.8 tok/s was measured over 143 tokens, so it isn't directly comparable.

## Recommended next steps

1. Test each WIP gate on its own against the warm DFlash control:
   - `GGML_METALIUM_KV_DIRECT_WRITE=1`
   - `GGML_METALIUM_KV_PAGED_ATTENTION=1`
   - `GGML_METALIUM_ELIDE_OUTPUT_COPIES=1`
   - `GGML_METALIUM_INTERNAL_FFN_SWIGLU=1`

   Each one must give exact output, zero fabric fallbacks, full batched-QKV coverage, a clean shutdown, and 100% JIT hits. Use the matching validator gate: `--require-direct-kv-write`, `--require-paged-attention`, `--require-output-store-elision`, or `--require-ffn-swiglu`.
2. Decide the precision policy before publishing N300 numbers.
   - Today N300 perplexity is 20-25% above the CPU. Q4_K is stored as BFLOAT4_B, and default fidelity is low.
   - With BFP8 weights plus `GGML_METALIUM_MATMUL_FP32_DEST_ACC=1 GGML_METALIUM_MATMUL_FP32_HIFI3=1 GGML_METALIUM_FA_FP32_ACC=1`, the N300 matches the CPU, at a cost of 0-8% decode.
   - Details: `KNOWLEDGE.md`, "Benchmark validity".
3. Fix the open crashes:
   - The row-fold `dim % 32` assert at `ggml-metalium.cpp:1695` kills every Qwen3.5/3.6 model. Reject the write in `supports_op` so it falls back to the CPU.
   - The intermittent `GGML_ASSERT(src_idx < buf_size)` at `ggml-metalium.cpp:877` in the traced server.
   - The gemma4 TTNN `slice` padded-shape fatal.
   - Falcon3-7B through llama-server: an L1 overflow during canonical-prompt prompt processing (SDPA decode or `update_cache`, depending on gates). The `n300-falcon3-7b` target is the repro.
4. Commit one proven optimization at a time. Record its exact counters and before/after timing in `NOTES.md`, and its durable constraints in `KNOWLEDGE.md`.
5. Rerun the LocalMaxxing targets with `localmaxxing-n300/bench.py` after each promoted change.

Next targets for the TT backend, in order: precision, crash fixes, MUL_MAT_ID/ARGSORT for MoE, then the Gated DeltaNet ops (`evidence/20261002-glimmer-v079/README.md`, "TT specialization roadmap").

## Benchmarks

The three harnesses are split by engine, and results from different engines are never mixed:

- `localmaxxing-n300/bench.py` covers llama.cpp/Metalium.
  - It runs `lmx` remote mode on the canonical `reasoning-v1` prompt (SHA-256 `9000edaadb16fc77ab78b39917245a03d4d7585c3ba770cb045a47a0e0683445`).
  - It records llama-server's own `usage`/`timings` through a local proxy, because lmx v0.1.43 doesn't send Verified-run evidence.
  - A greedy probe is compared against the CPU reference targets, and the output is a submission-ready `run.json`. It never submits.
- `localmaxxing-n300/tt_bench.py` runs Tenstorrent's own tt-metal v0.79.0 model code on the N300 (`tt_transformers`, `models/demos/gemma4`, and `models/demos/blackhole/qwen36` adapted to Wormhole) with the same prompt. Results, Wormhole adaptations, and the Glimmer port are in `localmaxxing-n300/README.md`. Examples: Qwen3-8B 38.4 tok/s, Qwen3.5-9B 24.2, Qwen3-14B 22.1, gemma-4-12B 15.6.
  - **Muse Glimmer 30B runs in tt_transformers at 12.9 tok/s, against 6.5 for this backend's `n300-glimmer-trace`.** The port is on tt-metal branch `glimmer-port`:
    - `0e4bb2d2cc` adds the Glimmer port: BFP4 MLP, BFP8 attention, weights loaded from the BF16 GGUF.
    - `90997774b2` adds the decode logits readback fix.
    - Its 256 greedy tokens match llama.cpp BF16.
    - Evidence and the optimization notes, including where the session paused, are in `evidence/20261003-glimmer-tt-port/`.
  - **Qwen3.8-27B runs at 10.2 tok/s** through qwen36, fitted to 24 GB by four wrapper knobs in `tt_bench.py` (no packed prefill gate/up copy, BFP4 down projections, 512-token prefill chunks, sliced MLP prefill). Teacher-forced against BF16 its greedy tokens agree 96.9% (Q4_K_M: 98.0%). Evidence: `evidence/20261003-qwen27b-n300/`.
  - **With speculative decoding it runs at up to 29.1 tok/s (DFlash2; 21.6 over 640 tokens).** MTP-1/2/3 with the checkpoint's MTP head give 16.8 / 20.4 / 23.7 (15.7 / 18.4 / 19.7 over 640). Text is identical to plain greedy: `tt_bench.py run Qwen/Qwen3.8-27B --spec dflash2|mtp3`, tt-metal branch `qwen36-n300-mtp`. Campaign plan, cross-platform matrix and status: `evidence/20261003-qwen27b-n300/SPECULATIVE.md`.
  - **Cross-platform toplines for Qwen3.8-27B** (best method each, 640 tokens / concurrent): RTX 4080 llama.cpp MTP-4 70.7 tok/s / 119 decode-sum at 8 streams; N300 DFlash2 21.6 / 48.8 at 7 batched streams; CPU DFlash2 16.3 / 27 at 8 streams. The 4080 numbers need the current-tree CUDA build `llama-upstream/build-cuda-cur`; the production `build-cuda` lacks these speculative types. Matrix: `evidence/20261003-qwen27b-n300/SPECULATIVE.md`.
  - **At a LocalMaxxing-style 128K window** (2026-10-04): 4080 MTP-4 36.7 tok/s, N300 MTP-3 18.5 (bfp8 KV + `--host-embedding`), CPU DFlash2 16.4. With the context actually filled to 64K / 100K: 4080 14.0 / 10.5, N300 MTP-3 14.8 / 14.7. N300 long prompts used to hang the device intermittently (Wormhole di/dt). `TT_MM_THROTTLE_PERF=5` fixes it (0 of 20 hung, against 10 of 20 without) and is now set by `tt_bench.py` for the 27B. Details: `evidence/20261003-qwen27b-n300/SPECULATIVE.md`.
- `/home/pullin/personal/tenstorrent/n300_benchmarks` covers the TT-vLLM production engine, plus llama.cpp parallel-stream, context, and correctness cases.

bench.py results on Glimmer 30B Q4_K_M (median of three; TT runs on v0.79):

| target | 256 tokens | 640 tokens | TTFT | notes |
|---|---:|---:|---:|---|
| `cpu-glimmer` (reference, `-b 64 -ub 64`) | 10.0 | - | 3.69 s | Glimmer reference probe |
| `cpu-glimmer-max` | 10.5 | 10.2 | 2.29 s | `--no-repack --load-mode none`, hugepages, 32 threads |
| `cpu-glimmer-dflash` | 57.9 | 29.9 | 2.33 s | 50% acceptance at 640 tokens |
| `n300-glimmer` (no trace) | 3.3 | - | 3.17 s | fabric 144,935/0 |
| `n300-glimmer-trace` | 6.5 | 6.3 | 4.67 s | |
| `n300-glimmer-dflash` | 50.5 | 18.4 | 3.25 s | 32% acceptance at 640 tokens |
| `cuda4080-glimmer` (production `serve.sh` flags) | 28.8 | - | 0.42 s | 14.96 GB VRAM |
| `cuda4080-glimmer-dflash` (48 layers, CPU draft) | 137.5 | 66.9 | 0.50 s | 47% acceptance at 640 tokens |

Other targets: `cpu-qwen3-8b` 20.5 tok/s, and `n300-qwen3-8b-dflash` 23.7 tok/s (19.9% acceptance on this prompt, vs 57.4 tok/s on the code prompt). All probes matched the CPU reference.

Two caveats:

- **Prompt echo:** Glimmer copies the whole `reasoning-v1` prompt (about 250 tokens) at the start of its reasoning. Speculative runs at 256 tokens are therefore inflated, so use the 640-token column.
- **Lossy N300 numbers:** the N300 results above come at 20-25% higher perplexity than CPU/CUDA (precision policy above).

TT prefill and TTFT are noisy. lmx's per-request nonce changes the prompt length, so a new partial-ubatch shape can JIT-compile during a timed request. Decode is unaffected.

LocalMaxxing lists `Tenstorrent P150`, not `Tenstorrent N300`. Don't submit N300 results until an N300 entry exists. `localmaxxing-n300/README.md` ("Status") records the allowlist and detector evidence.
