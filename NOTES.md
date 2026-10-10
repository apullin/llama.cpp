# Tenstorrent N300 Dual-ASIC Project Notes

Branch: tt-dual-asic
Started: 2026-08-12T20:11:41.301580
Last updated: 2026-10-02

Dated log. Entries before 2026-10-02 often refer to TT-Metal v0.74 (`13adda80`), which has since been removed. Current state is in `HANDOVER.md`, and the operating rules are in `KNOWLEDGE.md`.

## Milestone M1: Single-ASIC baseline reproduction

Command used:
TT_METAL_RUNTIME_ROOT=/home/pullin/personal/tenstorrent/generated/tt-metal-13adda80
llama-bench -m /home/pullin/personal/serving/Muse-Glimmer-30B-GGUF/muse-glimmer-30B-kquant-17gb.gguf --n-gpu-layers 20 --no-kv-offload 1 --n-prompt 256 --n-gen 64 -r 1 -o md

Results:
| test   | t/s |
|--------|-----|
| pp256  | 95.77 ± 0.00 |
| tg64   | 5.33 ± 0.00 |

Note: Brief expected ~28 / ~5.4 for single-ASIC. Measured pp256 is higher (95.77 tok/s). Decode ~5.33 tok/s matches expectation.

## Milestone M2: 1x2 mesh device open

Environment:
TT_METAL_RUNTIME_ROOT=/home/pullin/personal/tenstorrent/generated/tt-metal-13adda80
GGML_METALIUM_MESH_SHAPE=1x2

Command used for trivial workload:
llama-bench -m /home/pullin/personal/serving/Muse-Glimmer-30B-GGUF/muse-glimmer-30B-kquant-17gb.gguf --n-gpu-layers 1 --no-kv-offload 1 --n-prompt 1 --n-gen 1 -r 1 -o md

Observations:
- Device initialization logs show opening local chip ids/PCIe ids: 0/[0] and remote chip ids 1
- Logical multi-mesh adjacency: intra-mesh degree histograms mesh0 {1:2} indicates 2 devices in mesh
- Benchmark completed successfully with Metalium backend on 1x2 mesh
- Both ASICs responded; trivial workload executed without error

Next steps: proceed to M3 sharding.

## Milestone M3: Sharding design implementation

Date: 2026-08-12

- Added mesh device-count tracking.
- Added the 2-D weight sharding policy and post-matmul gather.
- The early shape-mismatch investigation and temporary assertion changes are archived in `SUPERSEDED.md`.
- The final correctness evidence starts in the next section.


## 2026-08-13 M3/M4 VERIFIED (Kimi takeover)

Gate results (prompt "The capital of France is", temp 0, n 20):
- CPU -ngl 0 vs single-ASIC -ngl 1: token-identical.
- Mesh 1x2 -ngl 1: token-identical to both. GATE PASS.
- Controlled pair -ngl 20: single-ASIC vs mesh 1x2 token-identical
  ("...The capital of France is\nWe need to answer."). Sharding is
  numerically transparent.
- M4 -ngl 99 full offload on mesh: runs, coherent ("...Probably answer:
  Paris."). ~1.4 t/s decode, dominated by host-roundtrip gathers.

Root causes fixed today:
1. Blanket sharding caught token_embd (embedding kernel needs full vocab)
   -> name whitelist: only MUL_MAT-consumed weights shard.
2. This tt-metal build reports per-device SHARD logical_shape and
   REPLICATED mesh layout even for mapper-sharded tensors. All the
   "mesh shape mismatch" aborts were that misread. Sharding is now
   tracked by an explicit m3_sharded flag on the tensor meta.
3. Fused lowerings (ActLowering, LinearLowering) emit their own
   matmul/linear and bypassed the gather -> shared ggml_metalium_m3_gather.
4. CCL all_gather wedges the command queue on this N300 (single inter-chip
   link) regardless of mesh graph descriptor / topology setting. Gather is
   a host roundtrip (aggregate_tensor + concat + re-upload). CCL path
   remains behind GGML_METALIUM_M3_CCL=1.

Perf: mesh -ngl 20 = 3.3 t/s vs single-ASIC 5.5 t/s (gather tax exceeds
second-ASIC gain at partial offload). -ngl 99 mesh = 1.4 t/s (only way to
full-offload; single-ASIC OOMs). 4080 bar: 29 t/s. Fabric CCL is the lever.


## M5 numbers (2026-08-13, llama-bench -p 64 -n 64 -nkvo 1)

| config | pp t/s | tg t/s |
|---|---|---|
| CPU 32t (reference) | 88.7 | 10.6 |
| RTX 4080 (-ngl 48) | ~276 | ~29 |
| Metalium single-ASIC (-ngl 20) | 95.8 | 5.33 |
| Metalium mesh 1x2 -ngl 99 (shard+host-gather) | 16.6 | 1.38 |
| Metalium dual-ASIC layer-split -ngl 99 (GGML_METALIUM_DEVICE_ID=0,1) | 84.4 | 2.47 |

Layer-split = one 2x1 parent mesh + 1x1 submesh per ASIC, llama.cpp native
multi-backend layer split. No fabric/CCL. Output token-identical to mesh
full-offload run. Trace mode: captures but never replays (per-token graph
keys differ — KV length is baked into graph shapes; replay needs a
shape-stable padded decode graph).


## 2026-08-13 evening: CCL fixed + perf diagnosis (Kimi)

- CCL all_gather works on BOTH the pinned 13adda80 and upstream-main
  tt-metal. The missing call was tt::tt_fabric::SetFabricConfig(FABRIC_1D)
  BEFORE opening any device (backend does this now when a mesh shape is
  requested). Without it the fabric is armed "for dispatch only" and CCL
  kernels never complete. Commit 4a45908.
- tt-metal upstream-main build lives at
  ~/personal/tenstorrent/generated/tt-metal-upstream (built OK; needs
  system libssl-dev + libopenmpi-dev, toolbin clang-20, ninja, and a
  CMake RPATH cache seed). Backend port to it is STARTED (build-ttnew):
  Tensor moved to ttnn:: namespace (sed done, compiles both trees), but
  the custom device ops (embedding fold/gather, softmax, wkv7) hit the
  new device-op API and still need porting. Only finish if a feature
  there proves worth it.
- Perf: mesh+CCL = 2.4 t/s, layer-split = 2.47 t/s, mesh+host-gather =
  1.38 t/s. All modes converge ~2.5 t/s => the gather was never the main
  cost. Host profile (GGML_METALIUM_PROF=1): node-loop host time is only
  ~20% of wall (MUL_MAT ~35us host-side). Device-side execution dominates.
- Probe: default-config decode matmul [1,6656]x[6656,19968]^T bf16 takes
  2.03ms (~131GB/s = 45% DRAM peak). Decode-optimized program configs
  require L1-sharded activations (width-sharded L1 residency + multicast
  matmul configs, per tt-metal LLM demo idiom). That conversion is the
  next big lever.
- Trace mode captures but never replays: per-token graph keys differ
  (KV length baked into graph shapes). Replay needs a shape-stable
  padded decode graph.


## 2026-08-14: decode matmul probe matrix + [K,N] weight storage (Kimi)

Probe matrix (/tmp/mesh_probe/probe_mmperf.cpp, single ASIC, M=1 decode shapes,
20 iters after warmup). "default+T" = what the backend used to do
(weight [N,K], transpose_b=true), "default" = weight stored [K,N],
"dramshard" = demo DRAM-sharded idiom (weight DRAM-width-sharded over the 12
banks, activation padded to M=32 and L1-width-sharded over 16 cores,
MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig):

| shape (K x N) | dtype | default+T | default | dramshard |
|---|---|---|---|---|
| 6656x19968 | bf16 | 2.04 ms | 1.45 ms | 1.08 ms (86% DRAM peak) |
| 6656x19968 | bfp4 | 0.59 ms | **0.42 ms** | 0.71 ms (WORSE) |
| 6656x6656  | bf16 | 0.59 ms | 0.47 ms | n/a (N%12 banks != 0) |
| 6656x6656  | bfp4 | 0.156 ms | **0.149 ms** | n/a |

Conclusions:
- Weights ARE quantized on device (Q4_K -> BFLOAT4_B at tilize) and ttnn
  matmul dequants bfp4 natively in-kernel: 3.4x faster than bf16.
- DRAM-sharding LOSES for bfp4: quantized weight traffic is already small,
  so the matmul is compute-bound and the dram-sharded kernel uses fewer
  cores. The demo idiom is tuned for bf16/bfp8 weight streaming.
- The free win: store weights [K,N] and drop transpose_b (1.4x on every
  matmul). transpose_b forces strided DRAM reads.
- DRAM-sharded grid recipe that works on this N300 (for the record):
  cores = largest c<=64 with (K/32)%c==0 && (N/32)%c==0 that factors
  cols*rows with both <=8 (K=6656,N=19968 -> 16 cores as 8x2);
  weight ShardSpec over the 12 DRAM-bank cores [K, N/12] ROW_MAJOR requires
  N % (32*12) == 0; activation must be padded to a full M=32 tile row
  before L1 width-sharding or ShardSpec alignment fatals.
- bfp4 transpose on TILE layout dequants+requants (roundtrip err 0.109) —
  NEVER transpose quantized tiles. Transpose the row-major tensor pre-tilize
  (exact for bf16/f32), then tilize to bfp4 as the single quantization step.

Backend change (tt-dual-asic): whitelist MUL_MAT-only weights are transposed
[1,1,N,K] -> [1,1,K,N] on the row-major device tensor at upload, before
tilize; flag ggml_tensor_extra_metalium::weight_kn; the three matmul
emission sites (mul_mat, LinearLowering, ActLowering) pass
transpose_b = !weight_kn. Kill switch: GGML_METALIUM_NO_WEIGHT_KN=1.
Mesh sharding unaffected (shard along dim 2 happens before the transpose).


## 2026-08-14: where decode time goes (post-[K,N], layer-split -ngl 99)

Profiler: GGML_METALIUM_PROF=1 (host/op), PROF_SYNC=1 (adds device sync per op),
PROF_NAMES=1 (per-node keys; some ggml name fields are unterminated/garbage —
keys sanitized but junk names remain junk).

- Wall 312 ms/tok (8-tok run, 3.2 t/s). Host dispatch ~102 ms/tok
  (MUL 41us, ADD 113us, RMS_NORM 78us per call — slow ttnn dispatch).
- Small-op probe (probe_smallops.cpp): mul/add ~35us, silu 28us,
  rms_norm 145us pipelined; 10x mul chain = 32us/mul — no inter-op gap
  beyond flat per-op cost. ~1200 small ops/tok ≈ 40 ms/tok.
- Compute-config sweep (probe_ck.cpp): HiFi4 vs HiFi2 vs LoFi, fp32 acc
  on/off — ALL WITHIN NOISE for bfp4 and bf16 decode matmuls. Matmuls are
  at their floor; fidelity is NOT the lever.
- Matmul estimate ~110-150 ms/tok. Residual ~100 ms/tok = per-op dispatch
  of many distinct programs + non-op host time. Structural fix = trace
  replay (blocked: graph keys embed KV length; needs shape-stable padded
  decode graph).
- KNOWN LIMIT: -ngl 99 dual-ASIC OOMs at pp512 (DRAM banks 100% full after
  weights+KV; the 20MB [512,19968] FFN intermediate doesn't fit). pp64
  fine. Predates the [K,N] change (no memory delta). Mitigations: fewer
  ngl, activation release scheduling, smaller ubatch.


## 2026-08-15: lower-peak `[K,N]` upload + attention-gate overlap

### Upload-memory blocker and fix

The first attention-gate placement attempt was mathematically plausible but
could not load the model. `output.weight` is the largest quantized matrix.
The old upload path materialized its full device `[N,K]` tensor and then a
second approximately 5.38 GB device `[K,N]` transpose before releasing the
first. On the already-tight two-ASIC layout, the temporary allocation missed
by about 26 MiB in the largest free bank.

`quantized_ggml_data_to_tt_host_buffer_kn` now dequantizes eligible large
quantized weights directly into host `[K,N]` order. Upload then performs the
single tilize/quantize step into the final device layout and never allocates
the full device-side transpose source. This is token-exact for Glimmer Q4_K
and eliminates the peak that blocked `output.weight`; it still consumes about
5.38 GB of temporary host FP32 storage and has not been validated across all
GGML quant formats.

### Attention-gate execution design

- The normal llama.cpp 27/25 layer partition remains intact.
- Fifty `.attn_gate.weight` tensors live on the peer ASIC.
- `blk.25.attn_gate.weight` and `blk.26.attn_gate.weight` remain local so each
  direction moves 25 gates and device memory stays balanced.
- For M=1 decode, the peer gate matmul and its fabric transfers overlap the
  merged local Q/K/V descriptor program.
- For prompt/multirow graphs, the ordinary scheduler path uses the backend's
  direct cross-device tensor copy.
- `ActLowering` leaves remote gate matmuls visible to the overlap executor,
  but the two local exceptions retain normal fused activation lowering.
- Gate placement and lowering require both
  `GGML_METALIUM_INTERNAL_ATTN_GATE_PARALLEL=1` and
  `GGML_METALIUM_FABRIC_COPY=1`. The QKV planner additionally requires live
  fabric links, preventing a null-link dereference under partial setup.

The hard-coded local layer names are specific to Glimmer-52 with the default
split. Generalize or parameterize placement before enabling this mode for
another architecture.

### Correctness

Post-gate-guard smoke (superseded by the final geometry-hardening smoke below):

```text
The capital of France is
We need answer. Probably just Paris. Possibly user
[metalium-fabric-copy] direct=4916 fallback=0
```

Log:
`/tmp/llama-tt-check-ffn-attngate-final-guarded-20260815.log`

SHA256:
`7238c8fe76619857cb1b86365fa0ee0c8dd88a992f7b71247d4e6916c0672db5`

### Definitive A/B

Both modes used `-p 64 -n 16 -r 3`, one trace per ASIC, and zero fabric
fallbacks. `run.sh` now accepts `LLAMA_BENCH_REPS` and defaults it to 1.

| mode | pp64 | tg16 |
|---|---:|---:|
| dual trace + internal FFN control | 195.32 ± 2.04 | 6.72 ± 0.08 |
| attention gate, pre-guard build | 193.26 ± 1.66 | 6.86 ± 0.09 |
| attention gate, final retained build | 192.54 ± 1.93 | 6.84 ± 0.08 |

The retained decode gain is about 1.8 percent. Prompt throughput is slightly
lower but within run-to-run noise. The final control was captured before the
guard/local-fusion rebuild; both changes are gated by the attention-gate
environment variable and are inert in the control mode.

Evidence:

- `/tmp/llama-kv-tensor-trace-ffnparallel-attngate-r3-final-20260815.log`
  (`461212319ab2f802d7299bfe5d7f4fdbc28b97991e2731acc59979df6fc335bc`)
- `/tmp/llama-kv-tensor-trace-ffnparallel-dualtrace-r3-control-20260815.log`
  (`2178d8dd7c2c9a89b6dbe358b46a0b3ebc5040721b3cbb34d510e2bfd9bb8136`)

Trace capture headroom remains tight: about 258 MB free on the smaller side.
A 48-core Q / 64-core gate layout was token-exact but neutral at 6.78 tok/s,
so it was removed.

### Speculative audit after dual trace

| Dynamic Flash configuration | rate | acceptance |
|---|---:|---:|
| old path, draft max 7 | 5.656 tok/s | 17.5% |
| dual trace + internal FFN, draft max 7 | 6.136 tok/s | 17.5% |
| dual trace + internal FFN, draft max 3 | 4.993 tok/s | 33% |

Current drafting and graph overhead still lose to 6.84 tok/s dense decode.
The ideal lookup experiment (100% acceptance) reached about 134 tok/s, so the
32-stream throughput can only become one-stream throughput if several future
tokens are predicted accurately and verified together. A compatible trained
draft, MTP/Medusa-style head, or tree-verification path is more promising than
further tuning the current low-acceptance Dynamic Flash draft.


## 2026-08-15: FFN geometry screen and override hardening

The existing FFN environment knobs were screened on the retained attention-
gate path with `-r 1`. Results are same-session screening values, not precision
benchmarks:

| grid / per-core N / block width | tg16 | outcome |
|---|---:|---|
| 8x4 / 20 / 4 | 6.76 | retained default |
| 8x5 / 16 / 4 | 6.72 | rejected |
| 8x4 / 20 / 8 | 6.65 | rejected |
| 8x4 / 20 / 13 | 6.62 | rejected |
| 8x6 / 13 / 4 | 6.71 | rejected after hardening |

The first 8x6/N13 attempt aborted because `out_block_w=13` was paired with a
fixed `out_subblock_w=4`; TTNN requires exact divisibility. The program-config
builder now selects the largest subblock divisor no greater than four. It also
checks grid bounds, per-core N, K-block divisibility, and total output coverage;
invalid overrides log once and fall back to 8x4/N20/block-width-4.

Final post-hardening exact smoke:

`/tmp/llama-tt-check-ffn-attngate-final-hardened-20260815.log`

SHA256:
`e6e8397c67c645c91ecbadcb4947b741d22e9efb87cae5aa55a8a553344d48d7`

## 2026-08-15: Qwen3 tree-speculative single-stream frontier

### Retained result

The strongest verified path is a Qwen3-8B Q8 target on N300 with a Qwen3-0.6B Q4_K_M CPU draft. The target remains greedy. The draft sampler alone uses temperature 0.8. The tree uses two branches, a six-node budget, split threshold 0.15, top-k 2, unified KV, FP32 destination accumulation with HiFi3, and target-only peer-ASIC FFN overlap.

| configuration | decoded tok/s | logged acceptance | correctness |
|---|---:|---:|---|
| Q8 draft, fresh promoted precursor | 34.858 / 34.965 | 82.677% | exact prefix |
| Q4_K_M draft, top-k 10 | 37.061 / 36.992 | 85.366% | exact prefix |
| Q4_K_M draft, top-k 2 | 37.448 / 36.896 | 85.366% | exact prefix |
| promoted `qwen3-tree-tt-best` | 36.990 | 85.366% | exact prefix |
| depth-7 candidate | 37.618 | 78.832% | exact 470-byte prefix |

The two-run retained top-k-2 mean is 37.172 tok/s. Reducing top-k from 10 to 2 leaves the two-branch tree and acceptance counts unchanged while removing candidates that cannot be consumed; the measured gain is small and near run variance. The material gain is the Q4_K_M draft, whose branch ranking aligns better with the greedy Q8 target than the Q8 draft on this prompt.

`./run.sh qwen3-tree-tt-best` fixes every critical setting and writes `/tmp/qwen3-tree-tt.pid` plus `/tmp/qwen3-tree-tt.log`.

### Correctness gate

The target reference is `evidence/20260815-qwen-speculative/qwen3-target-q8-greedy-n200-reference.log`. For speculative logs, remove the interleaved `tree draft sampler temp` timestamp diagnostic, extract generation from the fixed prompt through the first metrics line, and require that it is a byte-exact prefix of the reference. The promoted depth-6 runs each produced a 467-byte exact prefix; the depth-7 candidate produced a longer 470-byte exact prefix.

`QWEN_TREE_TEMP` controls the target and must remain 0 for this gate. `LLAMA_SPEC_TREE_DRAFT_TEMP` controls only the draft sampler. Two earlier BF16 experiments accidentally set target temperature 0.8 and are not valid evidence.

### Fair hardware comparison

| Qwen3-8B Q8 path | tok/s |
|---|---:|
| N300 dense greedy | 12.62 |
| N300 retained tree | 37.172 mean |
| RTX 4080 dense greedy | 75.32 |
| RTX 4080 tree | 163.15 |

Tree verification gives N300 a real 2.9x gain over its dense path by converting efficient M>1 work into accepted future tokens. It still trails the 4080 dense path by roughly 2x and the CUDA tree path by roughly 4.4x.

### Rejected draft and sampler variants

- Draft temperature 0.6: 35.389 tok/s, 77.612% logged acceptance.
- Draft temperature 1.0: 36.934 tok/s, neutral.
- Split threshold 0.10 with Q4_K_M: 36.585 tok/s.
- Split 0.20: 36.929 tok/s with an identical target tree; split 0.30: 36.643 tok/s with the same target-row count and more draft calls.
- Draft budget 5: 36.377 tok/s. It reduced target rows from 171 to 165 but needed 26 target calls instead of 24.
- Draft budget 7: one exact run at 37.618 tok/s. It is a candidate, not promoted, because the confirmation hit a pre-load ETH-heartbeat failure.
- 48 draft generation/batch threads: 36.912 tok/s, neutral end-to-end. A CPU-only screen peaked at 255.70 tok/s with 32 physical-core threads; 40 and 48 threads fell to 251.86 and 249.31 tok/s.
- Community Q4_0: 32.764 tok/s; exact-source Q8-to-Q4_0 requant: 34.813 tok/s.
- Community Q2_K: 21.118 tok/s despite 312 tok/s isolated CPU decode; branch acceptance fell to 49.180%.

Draft throughput alone is therefore not predictive. Candidate ranking and target-token coverage dominate.

### Remote-Q and Eagle3 conclusions

Remote-Q was fully removed from source. BFP8 remote Q produced an incorrect greedy prefix at about 30.5 tok/s. Corrected BF16 remote Q was exact at 34.660 tok/s, versus 34.858 tok/s for matched local Q, so it added fabric/memory complexity without speed.

Eagle3 remains rejected. A 22.397 tok/s run accepted an invalid prefix; full-precision variants were roughly 12 tok/s. Do not repeat that sweep without a new correctness hypothesis.

### Timing instrumentation and depth-7 candidate

`LLAMA_SPEC_TREE_TIMING=1` now separates the speculative loop into acceptance/sampling/output, KV reconciliation, draft seed decode, tree CPU/bookkeeping, tree draft decode, target KV preparation, target decode, and residual time.

The exact depth-7 candidate decoded 132 tokens at 37.618 tok/s:

| component | time | wall share |
|---|---:|---:|
| target decode | 2408.238 ms | 68.6% |
| tree draft decode | 676.081 ms | 19.3% |
| tree CPU/bookkeeping | 168.486 ms | 4.8% |
| draft seed decode | 131.447 ms | 3.7% |
| accept/sample/output | 121.535 ms | 3.5% |
| target KV preparation | 1.129 ms | effectively zero |

It made 23 target calls over 188 target rows. Depth 5 made 26 calls over 165 rows and target decode rose to 2625.942 ms. Per-call target latency stayed roughly 101–105 ms across depths 5–7, so accepted output per target call matters more than row count alone. Split thresholds 0.15 and 0.20 produced the same target tree; 0.30 changed draft work without reducing the fixed target budget.

KV copying is not a useful target. Depth 8 is the next focused budget test once startup reliability allows it. Perfectly hiding all non-target work caps the depth-7 sample near 54.8 tok/s; further gains beyond that require faster target kernels.

Late in the sweep, ETH-heartbeat startup failures became frequent. The depth-7 candidate closed cleanly, but its confirmation failed before model loading. The exact PID exited, `fuser` showed no owner, and no second reset loop was attempted.

### Compiled next experiment: one-tile batched QKV merge

Metalium rounds floating tensor dimensions to 32x32 tiles. Verification batches with M=2..32 therefore occupy the same one-tile M extent, while the merged Q/K/V planner previously rejected every graph unless the logical row count was exactly one.

`GGML_METALIUM_QKV_PARALLEL_BATCHED=1` now opt-in admits Q/K/V inputs with 2..32 logical rows and `ne[2]=ne[3]=1` to the existing `per_core_M=1` descriptor merge. The ordinary-operation path remains the fallback if descriptor creation throws. Attention-gate overlap remains single-row-only.

The code compiles and the runner is:

```bash
LLAMA_SPEC_TREE_TIMING=1 ./run.sh qwen3-tree-tt-d7-qkvbatch-candidate
```

It is hardware-unverified because the confirmation sequence stopped after the repeated ETH-heartbeat failure. Default and promoted modes explicitly unset the environment flag. Require a byte-exact greedy prefix before interpreting any speed.

## 2026-08-16: depth-10 batched-QKV promotion

This section supersedes the 2026-08-15 retained result and compiled experiment above.

| configuration | warm decoded tok/s | target calls / rows | disposition |
|---|---:|---:|---|
| depth 8, ordinary QKV | 39.178 / 38.958 | 21 / 194 | exact control |
| depth 8, batched QKV | 39.987 / 39.777 | 21 / 194 | exact; QKV retained |
| depth 9, batched QKV | 40.294 / 39.915 | 20 / 207 | exact; superseded |
| depth 10, batched QKV | 41.426 / 41.709 | 19 / 214 | exact; promoted |
| depth 11, batched QKV | 40.007 | 19 / 234 | exact; rejected |

At matched depth 8, the one-tile QKV merge improved the two-run mean from 39.068 to 39.882 tok/s, about 2.1%. Both candidate runs reported every M>1 group fused. Depth 10 then improved accepted output per target call and raised the mean to 41.568 tok/s.

The final public mode reproduced 41.631 tok/s and is:

```bash
./run.sh qwen3-tree-tt-best
python3 validate_qwen_tree.py --require-qkv-batched /tmp/qwen3-tree-tt.log
```

The second warm depth-10 run spent 3.189 seconds in generation:

| component | time | wall share |
|---|---:|---:|
| target decode | 1998.858 ms | 62.7% |
| tree draft decode | 760.036 ms | 23.8% |
| tree CPU/bookkeeping | 192.350 ms | 6.0% |
| draft seed decode | 109.750 ms | 3.4% |
| accept/sample/output | 125.110 ms | 3.9% |
| target KV preparation | 0.896 ms | effectively zero |

First-seen verifier shapes compiled new JIT artifacts and produced misleading 20–25 tok/s cold runs. Compare only runs whose JIT summary reports 100% hits. Those cold logs are retained as negative measurement evidence.

The backend now reports `[metalium-qkv-batched] attempts=... fused=... logical_rows=...`; the promoted run reported 756/756 groups and 8,316 logical rows. Correct output, zero fabric fallback, clean teardown, and complete QKV fusion are all hard gates.

The next direct backend lever is decoder-chain fusion around RMS/scale, residual, activation, and multiply operations. The next speculative lever is safe overlap or caching of CPU draft work while the target verifies the current tree.

## 2026-09-27 upstream rebase

- Rebased the committed `tt-dual-asic` series from `84e908c` onto upstream `4da6337`.
- Replayed 21 local commits without content conflicts. Dropped local server-context commit `4e896e4`; upstream `dfc29b64e` now owns the YaRN effective-context policy.
- Restored the pre-rebase working tree from `stash@{0}`. Ported the DFlash alternate-path row gather onto upstream fused injection and M-RoPE handling. Upstream's newer DFlash output-head implementation replaced the old local one-line load.
- CPU build passed for `llama-completion`, `llama-speculative-simple`, and `llama-speculative` with `-j 64`.
- CPU direct-completion smoke (`-no-cnv`, temperature 0, 20 tokens) produced: ` Paris. It is the most populous city in France and the most visited city in the world.`
- The full Metalium-enabled build passed after adapting `graph_optimize` to the upstream callback signature.
- Initial 1x2 and single-ASIC smokes stopped before model load on `ETH core heartbeat check failed` for `e1-6` (post code `2040000`). After explicit approval, `tt-smi -r` restored the card.
- Post-reset 1x2 `-ngl 1` completion exited 0, matched the CPU output byte-for-byte, and decoded at 9.07 tok/s. Idle telemetry after close was 500 MHz, 14 W / 12 W, and 47.8 C / 40.0 C.
- See `SUPERSEDED.md` for the graveyard, backup refs, and exact replacement commits.

## 2026-09-28 TT-Metal v0.79 compatibility

- Added source compatibility for pinned TT-Metal `13adda80` and official stable `v0.79.0` (`de546d3b14`). Both Metalium builds complete with `-j 64`.
- Ported tensor namespace changes, host-storage access, device-tensor creation, SFPI conversions, and explicit v0.79 link dependencies without changing the v0.74 runtime path.
- Clean v0.74 1x2 `-ngl 1` smoke exited 0, produced the CPU continuation byte-for-byte, reached 9.10 tok/s, and reported 74/74 JIT hits.
- Clean v0.79 1x2 `-ngl 1` smoke passed twice with the same CPU continuation. The warm run reached 9.44 tok/s with 83/83 JIT hits.
- Full v0.79 Glimmer offload from the staged compatibility patch exited 0 and repeated the prior v0.79 continuation. The warm non-traced run reached 3.88 tok/s, reported 346/348 JIT hits, and used 2,205 direct fabric copies with zero fallback.
- The full-offload continuation changes after several tokens between v0.74 and v0.79, but repeats exactly within v0.79. Keep runtime-version results separate and require same-runtime controls for optimization gates.
- Switching directly from v0.79 to v0.74 left incompatible fabric firmware state and timed out during router sync. One scoped N300 reset restored a clean v0.74 run.

## 2026-09-28 Fabric teardown at exit (ETH heartbeat failures)

- **Symptom:** after a clean exit, the next open intermittently failed in UMD topology discovery with "Timed out waiting for ETH heartbeat ... e1-6" (local chip) or "... e1-0" (remote chip). Two of three re-opens failed on 2026-09-28.
- **Cause:**
  - The backend closed only the 2x1 parent mesh. The per-ASIC submeshes share its `ScopedDevices`, so `DeviceManager::close_devices` never ran. gdb saw only `MetalContext::teardown`, at static destruction.
  - The fabric routers therefore stayed `READY_FOR_TRAFFIC` on link cores L ch8/ch9 and R ch0/ch1.
  - The second-link workload router (`GGML_METALIUM_FABRIC_COPY=1`) yields to base firmware only about every 64 ms because of the default 10,000-iteration switch interval. That is longer than UMD's 50 ms heartbeat window.
- **Fix:** close submeshes before the parent at exit. Evidence after the fix:
  - gdb shows `DeviceManager::close_devices` and `FabricFirmwareInitializer::teardown` running.
  - All four link routers report `TERMINATED`, and heartbeat gaps stay under 0.4 ms.
  - Three N300 benchmark servers (DFlash, DFlash, Glimmer) and two staged-build completions re-opened back to back without a reset. The completions repeated the v0.79 continuation.
  - The staged patch builds against v0.74 and v0.79.

## 2026-10-02 v0.79 only

- **Removed:** TT-Metal v0.74. That covers the `generated/tt-metal-13adda80` worktree, the `llama-upstream/build` and `llama-metalium/` builds made against it, and the backend compatibility layer (`67922c248`).
  - Its local changes stay in `repos/tt-metal` as branches `uint32-remainder-support` and `paged-update-cache-compact-cb`.
  - `run.sh`, `run-dflash.sh`, `run-dspark.sh`, the FFN sweep scripts, `serve-metalium.sh`, and the `n300_benchmarks` llama.cpp cases now use v0.79 and `build-tt079`.
  - The JAX/TT-XLA and Python-ttnn probe modes of `run.sh` were deleted, because they depended on the removed runtime.
- **DFlash control:** warm v0.79 run on the code prompt, `QWEN_DFLASH_N_PREDICT=128 ./run-dflash.sh tt`. The validator passed with an exact greedy prefix: 57.424 tok/s, 112/219 accepted, 16 target calls, 235 verifier rows, fabric 1,366/0, batched QKV 684/684, JIT 305/305 (`evidence/20261002-v079-only/`).

## 2026-10-02 benchmark sweep: Glimmer on v0.79, CPU max-out, model matrix

Evidence: `serving/evidence/20261002-glimmer-v079/README.md`. Logs are `e*`, `s*`, `t*` (N300), `c*` (CPU and 4080), and `m-*` (model matrix and perplexity).

- **Trace capture is back on v0.79.** Glimmer `-ngl 99`, llama-bench tg128:

  | configuration | tok/s |
  |---|---:|
  | dual-ASIC traced (`e01`) | 7.01 |
  | + attention-gate overlap (`e05`) | **7.18** |
  | + precision gates (`e08`) | 6.82 |
  | untraced, KV on device | 6.58 |
  | untraced, KV on CPU | 3.97 |

  Server path (`n300-glimmer-trace`): 6.5 tok/s over 256 tokens, 6.3 over 640.
- **Server abort, still open.** One `n300-glimmer-trace` server start (`runs/20261002T170717Z`) aborted with `GGML_ASSERT(src_idx < buf_size)` at `ggml-metalium.cpp:877` (slow-path host copy). The identical command passed on retry. The committed tree (`67922c248`) hits the same assert at once on a Falcon perplexity run.
- **Topology:**
  - One ASIC plus CPU is slower than the CPU alone. `-ngl 20` gives 6.57 tok/s, `-ngl 30` gives 5.56, and `-ngl 40` runs out of memory.
  - 1x2 mesh tensor parallel gives 3.42 tok/s.
- **Batch (llama-batched-bench, PP 64 / TG 32):** with the fabric copy, aggregate decode is 5.6 tok/s at 1 stream, 67 at 16, 102 at 32, and 109 at 64. The tuned CPU ties at 32 streams (101).
- **Speculation, measured over 640 tokens because of the prompt echo:**
  - N300 DFlash 18.4 tok/s (32% acceptance), n-gram 9.3.
  - Lookahead decoding 1.9-2.1 tok/s (CLI), slower than dense.
- **Accuracy.** N300 perplexity is 20-25% above the CPU on Glimmer and Falcon3-7B. The cause is default compute fidelity plus Q4_K weights stored as BFLOAT4_B.
  - With BFP8 weights (Q8_0 GGUF) and `MATMUL_FP32_DEST_ACC`/`MATMUL_FP32_HIFI3`/`FA_FP32_ACC`, Falcon matches the CPU (7.492 vs 7.529) at 14.75 tok/s.
- **CPU max-out (7975WX):**
  - The read ceiling is 196 GB/s (4 CCDs). Tuned Glimmer decode reaches 10.67 tok/s, 87% of its 12.3 tok/s ceiling.
  - `--no-repack` doubles prefill (85 -> 173 tok/s). Hugepages via `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` add 2.5% decode.
  - ZenDNN: Q8_0 prefill +38%, BF16 gives NaN. OpenBLAS: 2.6-5.6x slower. Q4_0: +5% decode for worse perplexity.
  - CPU DFlash runs Glimmer at 29.9 tok/s over 640 tokens. New targets: `cpu-glimmer-max`, `cpu-glimmer-dflash`.
- **RTX 4080:** `cuda4080-glimmer-dflash` gives 66.9 tok/s over 640 tokens (47% acceptance) with the production 48-layer split. llama-bench at `-ngl 51` gives 33.6 tok/s.
- **Model matrix (llama-bench):** Falcon3-7B runs at 15.9 tok/s on the N300 (CPU 37.3, 4080 134). Every qwen35/qwen35moe model aborts on the row-fold `dim % 32` assert, and gemma4 aborts in TTNN `slice`. Op gaps: MUL_MAT_ID, ARGSORT, SSM_CONV, GATED_DELTA_NET, SOFTPLUS, and multi-section ROPE.
- **Harness runs** (256 tokens), CPU / 4080 tok/s:
  - Falcon3-7B 36.7 / 130.8
  - Qwen3.5-9B 22.2 / 106.5
  - Qwen3.8-27B 7.8 / 22.4
  - Qwen3.6-35B-A3B 26.8 / 111.0
  - gemma-4-26B-A4B 36.1 / 96.8

  `n300-falcon3-7b` aborts on an L1 overflow during canonical-prompt prompt processing: in SDPA decode with the usual gates, in `update_cache` with minimal gates.

## 2026-10-02 batch scaling, dual-ASIC check, and the TT-native baseline

Evidence: `serving/evidence/20261002-glimmer-v079/README.md`, "Batch scaling, both ASICs, and Tenstorrent's own stack". Logs are `b*`, `s-*`, `l-*`, and `t-native-*`.

- **Lockstep batches** (llama-batched-bench, B = 1/2/4/8/16), aggregate decode tok/s:
  - N300 Glimmer: 5.64 / 10.49 / 19.85 / 35.36 / 54.56
  - CPU Glimmer: 10.38 / 19.52 / 32.31 / 38.86 / 71.58
  - CPU Falcon3-7B: 35.5 / 64.7 / 104.5 / 125.7 / 212.9
- **Both ASICs hold layers (27/26) and pipeline parallelism is enabled, but there is no overlap.**
  - At B=16 one ubatch gives 54.6 tok/s, two give 29.6, and four give 15.6.
  - Falcon on one ASIC is within 5% of two ASICs.
- **Serving** (llama-server, 16 slots, canonical prompt), aggregate decode tok/s at 1/2/4/8/16 concurrent:

  | engine | 256 tokens | 1,024 tokens |
  |---|---|---|
  | CPU | 10.5 / 19.0 / 31.4 / 37.0 / 58.4 | 10.2 / 18.0 / 29.1 / 34.7 / 54.6 |
  | 4080 | 28.9 / 55.1 / 98.0 / 120.5 / 190.2 | 28.2 / 53.3 / 95.1 / 116.4 / 183.2 |
  | N300 | 5.4 / 8.7 / 10.4 / 6.6 / 5.8 (collapses) | not run |

  The N300 server also crashes on prompt-state restore (`GGML_ASSERT(offset == 0)`, `ggml-metalium.cpp:3782`), so it needs `-cram 0 --no-cache-prompt`.
- **TT-native Qwen3-8B on the N300** (tt_transformers, uv env at `<tt-metal>/python_env`):
  - performance mode: 34.9 tok/s at batch 1, 873.9 tok/s at batch 32, 31.5 tok/s average over 4,096 tokens
  - accuracy mode: 29.1 tok/s
- **Same model through this backend:** 7.2 / 39.2 tok/s at batch 1 / 32. The CPU gets 20.7 / 160.2.
- **Qwen3.8-27B can't run on this card.** TT's Qwen3.5/3.6 implementation is Blackhole-only (27B needs P150x4). llama.cpp aborts at load (row-fold assert) and the model doesn't fit (26.1 GB vs 24.5 GB).

## 2026-10-03 disk cleanup

- **Removed:** the `generated/tt-metal-upstream` worktree (upstream main `4422e9811c`, the base of the abandoned August `build-ttnew` port). Its only local change, an MPI C++ link fix, is saved as `tenstorrent/evidence/tt-metal-upstream-4422e9811c-mpi-cxx.patch`.
- **Also removed:** the duplicate Qwen3-8B tt_transformers cache (`~/.cache/tt_transformers/Qwen/Qwen3-8B/N300/tensor_cache_instruct_bfp8`; runs read `N300/N300/`), and 418 August `/tmp` scratch files and directories (4.2 GB, mostly `mg_metalium4.log`). `/tmp` files that docs cite were kept.
- **Removed by parallel cleanup sessions:** every llama.cpp build directory (`build-cuda`, `build-cpu`, `build-tt079`, `build-rebase-check`, `build-ttnew`), the tt-metal `build_Release` and `.cpmcache`, `~/.cache/tt-metal-cache`, `~/.cache/uv`, and the container layers and uv packages that `tenstorrent/generated/toolbin` linked to (clang, ninja, ccache).
- **Rebuilt:**
  - `build-rebase-check` (CPU only, Release, native, from the working tree at `67922c248`). The ten `localmaxxing-n300/targets/cpu-*.toml` targets run its `llama-server`. Smoke: `bench.py run cpu-qwen3-8b` measured 21.4 tok/s, against the 21.9 baseline. It also replaces `build-cpu` for the `cpu` modes of `run-dflash.sh` and `run-dspark.sh` (now with `llama-speculative-simple`).
  - `build-cuda` at the production commit `fe57b804e`, from the detached worktree `build-cuda-src`; `--version` matches the old binary (build 2, `fe57b804e`). Smoke: llama-bench Qwen3-0.6B Q8_0 on the 4080, 562 tok/s tg32. The production server was already stopped and was not restarted.
  - tt-metal v0.79.0 `build_Release`, after repointing `toolbin` at the system clang 22, ninja, and ccache and adding CMake 3.31 (see `HANDOVER.md`). Smoke: opened the 1x2 mesh.
  - `build-tt079` against it. Smoke: llama-bench Qwen3-0.6B Q8_0 on the N300 Metalium backend, pp32 59 tok/s.

## 2026-10-03 Qwen3.8-27B on the N300 (TT-native qwen36)

Supersedes the 2026-10-02 "Qwen3.8-27B can't run on this card": that was the llama.cpp backend and the unmodified qwen36 formats. Evidence: `serving/evidence/20261003-qwen27b-n300/README.md`.

- **Fit:** the qwen36 TP path keeps a prefill-only packed `[gate|up]` copy of every MLP layer next to the decode `w1`/`w3` (3.2 GB per ASIC for the 27B). Dropping it and storing the down projections as BFP4 takes the weights from about 15.3 to about 10.6 GB per ASIC (`QWEN36_GATEUP_AGMM=0`, `QWEN36_DTYPE_OVERRIDES=mlp.down_proj=bfloat4_b`).
- **Prefill:** upstream sizes prefill for 2048-token chunks on 4-8 Blackhole devices. At TP=2 on Wormhole L1 the DeltaNet norm/concat outputs, the attention out-projection, and the unfused MLP gate/up matmul (3.9 MB of circular buffers per core) each overflowed. `QWEN36_PREFILL_CHUNK=512` and `QWEN36_MLP_PREFILL_ROWS=256` fix all three.
- **Result:** 10.2 tok/s decode, 748 ms TTFT for 306 prompt tokens. CPU Q4_K_M: 7.8; RTX 4080: 22.4; one P150 without speculation: 20.2 (Lottolabs).
- **Quality:** teacher-forced against BF16 on the CPU, 96.9% top-1 agreement over 256 greedy tokens, all in the top 5, mean BF16 NLL 0.108. llama.cpp Q4_K_M: 98.0%, NLL 0.214.
- **Next lever:** speculative decoding. Qwen3.8-27B ships a one-layer MTP head, but qwen36 drops MTP weights and has no multi-token verify step. The DeltaNet recurrent state would need snapshot/rollback for rejected drafts. Every faster P150 entry uses MTP or DFlash2.

## 2026-10-03 Qwen3.8-27B MTP-1 speculative decoding (TT-native)

Evidence and campaign plan: `serving/evidence/20261003-qwen27b-n300/SPECULATIVE.md`. Code: tt-metal branch `qwen36-n300-mtp` (`1cfa0467e3`, `a5e5f5e10e`), `models/demos/blackhole/qwen36/tt/spec/`.

- **Mechanism:** the checkpoint's one-layer MTP head drafts one token; a two-row verify step checks it. Full-attention rows alias one page-table row with per-row positions (two separate `paged_update_cache` calls, because rows in one cache tile race in a single call). DeltaNet layers batch the projections but run the conv and recurrence row by row, keeping the state after row 0 for rollback. Verify, MTP, and rollback are each traced.
- **Exactness:** verify-row logits and rollback state are bit-identical to plain decode. Text is identical to plain greedy over 256 and 640 tokens.
- **Speed** (`tt_bench.py --spec mtp1`): 15.0 tok/s at 256 tokens (127/129 accepted), 14.1 at 640 tokens (87% accepted), against 10.2 plain. A step is 132 ms: the verify is 125 ms, against 98 ms for a plain step, because the row-by-row DeltaNet chain doubles.
- **Memory:** 272 MiB free per ASIC with everything in place (outside the 1 GiB trace region).
- **Next:** batch the DeltaNet conv and q/k/v prep across rows; fold MTP into the verify trace with an on-device cross-shard argmax; MTP-k.

## 2026-10-03 Qwen3.8-27B MTP-2/MTP-3 and a cheaper verify step

tt-metal `qwen36-n300-mtp` commits `6f71490fba` (verify) and `8b33b4e80e` (MTP-k). Details: `serving/evidence/20261003-qwen27b-n300/SPECULATIVE.md`.

- **Verify cost:** all DeltaNet work except the recurrent state chain now runs once over all verify rows (conv window laid out one row per tile, q/k/v/beta/g prep batched, state kept in L1 between rows). A 2-row verify went from 123.7 to 110.0 ms (plain step 96.4); each extra row costs about 0.24 ms per DeltaNet layer. MTP-1 went from 15.0 to 16.8 tok/s.
- **MTP-k:** the single MTP layer is reused for every draft. Draft j reads draft j-1's token one position later plus the previous MTP output hidden after `mtp.norm` (vLLM `qwen3_5_mtp.py`; llama.cpp `qwen35.cpp` graph_mtp). After each verify, a catch-up MTP pass rewrites the MTP KV for the committed tokens. Rollback goes to the snapshot after the last accepted row.
- **tt_bench** (`--spec mtpK`), with text identical to plain greedy in every run: 256 tokens: MTP-1 16.8, MTP-2 20.4, MTP-3 23.7 tok/s. 640 tokens: 15.7, 18.4, 19.7 (MTP-3 accepts 3.08 tokens per step). Plain: 10.2.
- **Memory:** MTP-3 leaves 70 MiB free per ASIC with the default 1 GiB trace region, of which the traces use 34 MiB. `QWEN36_TRACE_REGION_MB=128` gives the same text and speed with 966 MiB free.

## 2026-10-03 Qwen3.8-27B DFlash2 on the N300 (TT-native)

tt-metal `qwen36-n300-mtp` commit `5ee07b70e6`; details in `serving/evidence/20261003-qwen27b-n300/SPECULATIVE.md` ("DFlash2 (checkpoint facts and step)").

- **Drafter:** z-lab/Qwen3.8-27B-DFlash2 on device (642 MiB per ASIC). It is fed by target hidden taps at layers 5/19/33/47/61 and drafts a block of 8 ([anchor, mask x7]) per step. An 8-row verify accepts the longest matching prefix. Drafter outputs track the official torch drafter at PCC 0.99.
- **Result** (`tt_bench.py --spec dflash2`): 29.1 tok/s at 256 tokens (5.78 tokens per step), 21.6 at 640 (4.30). Text identical to plain greedy. Best N300 method so far (MTP-3: 23.7 / 19.7).
- **Memory:** needs `QWEN36_TRACE_REGION_MB=128` (traces use 53 MiB); with it, 107 MiB free per ASIC. The opt-in host embedding table (`DFlash2Decoder(host_embedding=True)`) frees 1.2 GB per ASIC for about 1 ms per step.
- **Bottleneck:** the 8-row verify is 176 of each 198 ms step, because of the per-row DeltaNet state chain (about 10.8 ms per row). A fused multi-row recurrence kernel is the next lever for every speculative method.
- **Negative:** `ttnn.topk` over a 124K vocab shard runs single-core (36 ms); chunked top-k plus a host merge takes 1.7 ms. Acceptance at 640 tokens swings about 10% on tie-breaks.

## 2026-10-03 Qwen3.8-27B cross-platform toplines

Matrix and runs: `serving/evidence/20261003-qwen27b-n300/SPECULATIVE.md` ("Cross-platform matrix"). Best method per hardware, single stream at 640 tokens / concurrent aggregate:

- **RTX 4080** (llama.cpp current tree, `build-cuda-cur`, UD-Q4_K_M): MTP-4 70.7 tok/s (`-ngl 62 -ub 256`; re-run 69.8). Plain 37.2 at `-ngl 99`; the production fe57b80 build's 54-layer split gives 22.4. 8 streams: 119 tok/s decode sum, 92 wall-clock (`-ngl 60`).
- **N300** (TT-native qwen36): DFlash2 21.6 tok/s (exact greedy). 7 batched streams: 48.8 tok/s decode, bit-equal to B=1.
- **CPU** (llama.cpp, UD-Q4_K_M): DFlash2 n-max 4 16.3 tok/s (re-run 16.2). 8 streams: 27 tok/s decode sum, 22 wall-clock.
- llama.cpp speculative output isn't bit-exact against plain greedy (near-tie divergences around char 39-46 on the 4080; DFlash2 n-max 7 on the CPU). The N300 implementation is bit-exact.
- The DFlash2 drafter loses to MTP on the 4080: with the drafter on the GPU only 51 target layers fit; with it on the CPU the drafter is the bottleneck.

## 2026-10-04 Qwen3.8-27B at LocalMaxxing-style 128K windows and real depths

Details: `serving/evidence/20261003-qwen27b-n300/SPECULATIVE.md` (topline, decode-at-depth table, 2026-10-04 status entries).

- **LocalMaxxing context facts** (API, all 1,268 Qwen3.8-27B runs): 19% declare a window of 128K or more (36% of verified runs). Almost none measure with the context filled (61 runs have `prefillTokens` > 0).
- **128K window, canonical prompt, 640 tokens:** 4080 MTP-4 q4_0 KV 36.7 tok/s (70.7 with a small window), CPU DFlash2 16.4. N300 MTP-3 18.5 and plain 10.0, with bfp8 KV + host embedding; DFlash2 tops out at an 88K window (23.1).
- **Decode with the context filled** (16K / 32K / 64K / 100K):
  - 4080 MTP-4: 27.5 / 23.9 / 14.0 / 10.5.
  - 4080 plain: 18.9 / 16.7 / 13.1 / 10.8.
  - N300 MTP-3: 17.2 / 17.1 / 14.8 / 14.7.
  - N300 plain: 9.8 / 9.7 / 9.3 / 9.0.
  - CPU: 7.8 at 16K (DFlash2). Its prompt reading collapses past ~16K (~15 tok/s past 20K).
  - The N300 holds speed at depth because everything stays on the device. The 4080 has to offload layers to the CPU to fit the 128K KV.
- **Harness fixes:** `bench.py` now records `contextLength` (LocalMaxxing defaults it to 2048). `tt_bench.py --host-embedding` covers plain/MTP. N300 speculative decoders take multi-chunk prompts (`prefill_chunked_collect`). Verify rows use one B=1 SDPA each for exactness at depth (2-3% cost at short context).
- **Long-prefill hang, found and mitigated:** intermittent `device timeout in fetch queue wait` during 32K+ prefills (10 of 20 processes in an A/B). tt-triage plus tt-exalens show one Tensix core on the L chip (device 0) stalling mid-op in the DeltaNet out-projection matmul or the residual add after its reduce-scatter, almost always logical core (7,4). That matches Tenstorrent's Wormhole di/dt reports (tt-metal#12438, #13091, #16271). `TT_MM_THROTTLE_PERF=5` (NOPs in matmuls on >48 cores) brought it to 0 of 20 (agent 17 + parent 3). Prefill is 2-6% slower, decode unchanged, outputs identical. Now in `tt_bench.py` MODEL_ENV for the 27B and in the dev-harness defaults. Side finding: the L chip hits its 75 C limit during long prefills and the clock drops, so back-to-back prefills slow down by up to ~18%.
