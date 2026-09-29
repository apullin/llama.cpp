# Superseded work

This file records local patches, experiments, and conclusions that should not be restored after an upstream rebase.

## 2026-09-27 upstream rebase

- Old upstream base: `84e908c625fb60992b4cdef8180fb12fa9b4c4bf`
- New upstream base: `4da6337767f973e2b4d0797e5b323d77d8565e4a`
- Pre-rebase branch tip: `6b07b3c5898edcc7032cddd6524fff5878e3f5bb`
- Backup ref: `backup/tt-dual-asic-pre-rebase-20260927-220804`
- Dirty-tree stash: `pre-upstream-rebase 20260927-220804`
- External snapshot: `/home/pullin/personal/serving/tt-dual-asic-pre-rebase-20260927-220804`

### Dropped because upstream owns the behavior

- `4e896e4 server: allow ctx > training ctx when RoPE scaling is configured`
  - Not replayed during the rebase.
  - Upstream `dfc29b64e` moves YaRN effective-context handling into `llama_context`, adjusts `n_ctx_train`, and lets the server use the centralized value.
  - Do not restore the old server-local cap exception without first proving a remaining non-YaRN use case.

### Obsolete M3/M4 debugging states

These commits remain in history because later fixes were developed on top of them, but their conclusions and temporary assert changes are not current behavior:

- `c85f367` and `7950244`: temporary relaxed shape assertions.
- `d936802`, `f20563f`, and `9769503`: reports that the mesh shape mismatch or full offload were blocked.
- `62ff916`: treated a no-crash bench as a correctness gate. This was not sufficient.
- `67cc053`: recorded the old `llama-cli` chat/parser problem before the real deterministic comparison.

They are superseded by:

- `01c202b` (rebased as `48e042dbe`): token-identical 1x2 mesh output at `-ngl 1` and `-ngl 20`.
- `f749e0e` (rebased as `b42758b74`): full 52-layer `-ngl 99` mesh offload.
- `c2dbde2` (rebased as `3573ae620`): verified M3/M4 evidence and performance record.

The original `202048` versus `101024` mismatch was a missing gather. Current code uses `ggml_tensor_extra_metalium::m3_sharded` on the weight (`src0`) in `ggml_metalium_m3_gather()`. Do not restore operand-layout guessing.

### Upstream CLI changes

- Upstream `9fee29e94` removed the old `-no-cnv` compatibility argument from the CLI.
- The HTTP-based `llama-cli` is not the right raw-completion correctness harness. Use `llama-completion` for direct deterministic completion tests on current upstream.

### Reviewed and not superseded

Upstream `4da6337` has no `ggml/src/ggml-metalium` tree and no Metalium backend registration. GitHub searches for open or closed `Metalium`, `TTNN`, and `Tenstorrent` work found no competing backend PR. The following local work is still required:

- Base Metalium backend integration (`482225a76` after rebase).
- 1x2 weight sharding, output gather, and full-offload fixes.
- Dual-device layer split and fabric setup.
- `[K,N]` weight storage and Metalium-specific profiling/trace experiments.

Upstream tensor-split and meta-backend changes do not replace the Metalium mesh implementation because upstream has no Metalium backend to consume them.

## Build note after rebase

- CPU validation target on current upstream is `llama-completion`, not `llama-cli` when configuring without CURL.
- `cmake --build build-rebase-check --target llama-completion -j 64` passed.
- Rebuilt the missing pinned tt-metal `build_Release` with all 64 logical CPUs. Clang 22 was exposed under the pinned toolchain's `clang-20` names, and install required `CMAKE_BUILD_WITH_INSTALL_RPATH=ON` plus the build-library directories in `LD_LIBRARY_PATH`.
- The full Metalium-enabled llama.cpp build passed after updating the upstream `graph_optimize` callback signature.
- The first 1x2 device smoke stopped on a stale ETH heartbeat before model load. After explicit user approval, `tt-smi -r` restored the card.
- Post-reset 1x2 `-ngl 1` direct completion exited 0 and matched the CPU completion byte-for-byte. Decode measured 9.07 tok/s.

## Restored working-tree audit

- The stashed one-line DFlash `output.weight` load was not restored. Upstream `bf0a29cc1` now loads an optional draft output head with reduced-vocabulary support, and `633733d0a` adds token-embedding fallback. The upstream implementation fully supersedes the old line.
- Upstream DFlash2 and fused feature injection changed the same `common/speculative.cpp` path, but do not implement the saved alternate verifier tree or `process_path` API. That experiment was retained and ported to upstream's `batch_inject.embd` buffer and M-RoPE position handling.
- The saved Metalium direct-KV-write, paged-Q alignment, output-copy elision, and dual-ASIC projection work has no upstream Metalium equivalent and remains active work.
- `ggml-metalium.cpp.orig`, `generated/` inspector output, `test_goal_writeup.md`, and `typescript` are artifacts, not source changes. They remain recoverable in the external pre-rebase snapshot and should not be committed.

## TT-Metal v0.79 supersession

- The external v0.74 UINT32 remainder workaround is not part of the v0.79 port. Official v0.79 includes the relevant UINT32 shift and remainder fixes, including TT-Metal PRs 56682 and 55506. Keep the known-good v0.74 tree unchanged instead of replaying that patch into v0.79.
