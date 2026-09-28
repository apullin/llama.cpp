# Tenstorrent N300 / llama.cpp Metalium knowledge log

Last verified: 2026-08-19

This is the current operating guide. Historical milestones and failed paths
remain in `NOTES.md`.

## Pinned environment

- Runtime path: `TT_METAL_RUNTIME_ROOT=/home/pullin/personal/tenstorrent/generated/tt-metal-13adda80`
- Exact runtime state: `1f28bfa34b`, one local UINT32-remainder commit atop
  `13adda80c1`, plus the preserved compact paged-update output-CB diff. The
  built `_ttnncpp.so` includes that diff.
- Build: set `TT_METAL_HOME` to the same generated tree.
- Repository: `/home/pullin/personal/serving/llama-upstream`
- Branch/HEAD: `tt-dual-asic` at `6b07b3c5898edcc7032cddd6524fff5878e3f5bb`, plus an uncommitted Metalium patch.
- Hardware: Wormhole N300, two approximately 12 GB ASICs joined by one inter-chip link.
- Installed firmware bundle is 19.6.0; the pinned tree reports 19.5.0 as its newest fully tested bundle.

Preserve dirty state under `/home/pullin/personal/tenstorrent`. Do not reset,
clean, upgrade, or flash it as part of ordinary performance work.

## Safe device workflow

- Every TT workload must use a detached runner (`run.sh`, `run-dflash.sh`, or
  the equivalent exact-PID pattern) and write its PID and log under `/tmp`.
- Before a new workload, verify no owner with `fuser /dev/tenstorrent/0`.
- Never use broad `pkill -f`; do not touch `llama-server` or resident agents.
- `Read unexpected run_mailbox` can recover. If the recorded PID is alive,
  wait and inspect its log rather than resetting underneath it.
- For a genuine ETH-heartbeat startup failure, confirm the exact PID exited
  and preserve the log. The 2026-08-19 session stopped after failures on
  `e1-6` and then `e1-0`; do not reset or retry-loop in that session. A future
  explicitly healthy session may use at most one scoped recovery attempt.
- After an exception that began async fabric transfers, assume the device may
  need the same scoped recovery. Do not attempt a host fallback mid-transfer.

## Build and validation

```bash
cd /home/pullin/personal/serving
./run.sh llama-build
./run.sh llama-tt-check-ffn-attngate
LLAMA_BENCH_REPS=3 ./run.sh llama-kv-tensor-trace-ffnparallel-attngate
LLAMA_BENCH_REPS=3 ./run.sh llama-kv-tensor-trace-ffnparallel-dualtrace
QWEN_DFLASH_N_PREDICT=143 ./run-dflash.sh tt
./run.sh qwen3-tree-tt-best
```

The deterministic smoke must produce:

```text
The capital of France is
We need answer. Probably just Paris. Possibly user
```

Also require zero fabric fallbacks, one trace per ASIC in decode, and clean
device teardown. The retained r=3 gate/control rates are 6.84 ± 0.08 and
6.72 ± 0.08 tok/s.

For the promoted Qwen path, require
`QWEN_DFLASH_N_PREDICT=143 ./run-dflash.sh tt` followed by
`python3 validate_qwen_dflash.py --require-promoted-dflash --require-metalium-gates /tmp/qwen3-dflash-tt.log`.
The retained warm mean is 54.791 tok/s with an exact greedy prefix, 126/233
accepted drafts, 17 target calls over 250 rows, 720/720 batched-QKV fusion,
1438/0 fabric transfers, 301/301 JIT hits, and clean teardown. The older tree
mode remains a profiler control at a 41.568 tok/s warm mean.

## Architecture choices that worked

### Native layer split beats full mesh sharding

Full 1x2 mesh sharding works, but every partial matmul needs a gather. Even
with working CCL it converged near 2.4 tok/s; host gather was about 1.4 tok/s.
The useful topology is llama.cpp's native multi-backend layer split with one
1x1 submesh per ASIC, then explicit peer work only for independent projections.

This preserves one stable decode graph per layer partition and avoids the 53
trace states per ASIC created by a generic scheduler-level weight split.

### Store decode weights as `[K,N]`

Quantized bfp4 matmul is already efficient, but `transpose_b=true` creates
strided reads. Uploading weights in `[K,N]` order and using
`transpose_b=false` was the large general win. Never transpose quantized TILE
data: that dequantizes and requantizes. Transpose row-major data before the
single tilize/quantize step.

For very large quantized tensors, dequantize directly into a host `[K,N]`
buffer. This avoids holding both full device orientations at once. The path is
verified for Glimmer Q4_K `output.weight`, needs roughly 5.38 GB temporary
host FP32 memory, and still needs coverage for other quant formats/models.

### Keep graphs shape-stable and trace both ASICs

KV indices use I32 tensors with a host scalar shadow. Single-row updates stay
on device, avoiding readback and stabilizing the decode graph. Internal FFN
work uses a second command queue, so traces must be captured/replayed/released
on both ASICs. The known-good path has one trace state per ASIC.

Trace capture is memory-tight: the smaller side has only about 258 MB free.
Avoid experiments that add persistent outputs or many trace states.

### Send activations, not partial weight results

Persistent bidirectional MeshSocket pairs move the small layer activation to
the peer, execute an independent projection there, and return its output.
This is better suited to the single N300 link than sharding every matmul and
gathering large partial results.

`GGML_METALIUM_FABRIC_COPY=1` is a hard dependency for internal peer-weight
paths. Inspect `[metalium-fabric-copy] direct=... fallback=...`; any fallback
invalidates a performance result.

## Current parallel projection paths

### FFN

Each `.ffn_up.weight` is placed on the peer ASIC. The remote up projection
overlaps the local gate projection, then the up output returns over fabric.
This naming convention is relatively general across llama.cpp architectures,
but shapes and memory balance still need per-model validation.

### Glimmer attention gate

Fifty `.attn_gate.weight` tensors are placed on the peer. The M=1 gate matmul
overlaps merged local Q/K/V. `blk.25` and `blk.26` remain local to balance the
27/25 Glimmer layer split, and retain normal activation fusion.

This placement is not model-generic. The two hard-coded local names must be
parameterized or derived from the actual split before using the mode on a
different layer count. The QKV planner attaches a remote gate only when live
fabric links exist.

## Decode geometry lessons

- Q and Glimmer attention gate: 8x4 workers, per-core N 4, input block width 8.
- K and V: 4x1 workers, per-core N 2, input block width 8.
- Attention output: 8x7 workers, per-core N 4, input block width 4.
- FFN gate/up: 8x4 workers, per-core N 20, input block width 4.
- A 48-core Q / 64-core gate experiment was exact but neutral; more cores do
  not automatically mean faster bfp4 M=1 matmul.
- A coarse FFN screen retained 8x4 / per-core N 20 / block-width 4. Tested
  alternatives ranged from 6.62 to 6.72 tok/s versus a 6.76 same-session
  default screen.
- FFN override values are bounds/shape checked. The builder derives an output
  subblock width that divides per-core N; invalid combinations fall back to
  the default instead of reaching a TTNN fatal.
- DRAM-sharded demo kernels help BF16 streaming but lost for bfp4, where the
  default non-transposed kernel is closer to a compute floor.
- Broad HiFi/LoFi and fp32-accumulation sweeps did not expose a general speed lever. Precision changes can still alter target tokens or speculative acceptance, so test them only as scoped, exact end-to-end variants.

## Turning batch capacity into one-stream speed

M=1 decode underutilizes the machine and pays hundreds of distinct device programs per token. Batched work raises the effective M dimension and reached about 118.9 aggregate tok/s at 32 streams and 155.4 at 64 streams.

The verified bridge is DFlash block speculation. The retained mode is:

```bash
QWEN_DFLASH_N_PREDICT=143 ./run-dflash.sh tt
```

It uses a Qwen3-8B Q8 target on N300 and the official five-layer DFlash
block-16 sidecar on CPU. Draft depth 14 emits 143 output tokens, accepts
126/233 drafts, and needs 17 target calls over 250 verifier rows. Each target
batch has at most 15 logical rows and remains inside one 32-row Metalium tile.

Two retained warm runs measured 54.509 and 55.072 tok/s, a 54.791 mean. Dense
N300 Qwen is 12.62 tok/s, so DFlash is a 4.34x gain. The same target is 75.32
tok/s dense on the RTX 4080 and 343.059 tok/s with CUDA/CUDA DFlash. TT nearly
reaches the fair dense GPU bar but remains far behind the same speculative
algorithm on CUDA.

Correctness and warmth are hard gates. Require an exact byte prefix of the
retained greedy reference, consistent speculative counters, zero fabric
fallback, clean device shutdown, equality between batched-QKV attempts and
fused groups, and 100% JIT hits for a steady-state comparison. The exact
same-shape control with seven of 301 JIT misses measured only 35.269 tok/s.

Set `LLAMA_SPEC_DFLASH_TIMING=1` to print draft generation, target preparation,
target decode, speculative processing, sampling, commit/cleanup, and residual
wall time after the normal metrics. The flag is off by default.

The two warm runs spent 642.598 and 620.848 ms in DFlash generation and 99.97
and 93.00 ms in sampling. Roughly 1.88 seconds remained in target decode plus
loop/bookkeeping, or about 105-110 ms per target call. Perfectly hiding all
draft and sampler work would cap this sample near 76 tok/s. The primary lever
is reducing decoder-program and synchronization cost across all 17 calls.

A preserved inspector trace from the numerically invalid output-only HiFi4
run is still structurally informative. Its 26 target calls produced about
57,400 workload-commit events. Fused QKV, SDPA, scalar, transpose, unary, and
binary workloads each appeared exactly 936 times, or 36 layers x 26 calls,
which is roughly 61 workload events per layer/call. The inspector does not
provide exclusive timing by family, but it confirms heavy data-movement and
program fragmentation.

The strongest concrete fragmentation lead is repeated output storage. Generic
copy workloads 126 and 30 committed 4,232 and 3,816 times; workloads 129-132
added another 8,280, for 16,328 commits in those copy families. The backend's
value overload of `ggml_metalium_store_tensor` normally copies a compatible
new result into the prior allocation to keep its address stable. That address
contract is required by trace replay, but ordinary DFlash explicitly disables
tracing.

`GGML_METALIUM_ELIDE_OUTPUT_COPIES=1` is a built but device-unqualified
experiment. With tracing disabled it replaces only uniquely owned compatible
output handles; traced execution and shared aliases retain the old copy path.
It is off by default. `GGML_METALIUM_OUTPUT_STORE_STATS=1` reports the same
`[metalium-output-store]` counters without changing behavior. The validator's
`--require-output-store-elision` gate requires `trace=0`, `elide=1`, and at
least one replacement. Do not promote this path until an exact warm N300 run
shows a real target-side improvement and clean teardown.

`GGML_METALIUM_KV_DIRECT_WRITE=1` is a second, independent, default-off
experiment. For Qwen's `[1,1,M,1024]` tiled BF16 K/V output with `M <= 32`, it
replaces transpose, compact slice, L1 sharding, index typecast, and
`paged_update_cache` with one data-movement program that copies the two
physical face rows of each width tile directly into the indexed tiled cache
row. The first contract is intentionally narrow: same device, interleaved DRAM,
tiled BF16 source/cache, a one-stick row-major UINT32 index tensor, one padded
source row tile, tile-aligned cache rows, cache names `cache_k_l*` or
`cache_v_l*`, and tracing disabled. Ineligible tensors keep the proven fallback.

Program-cache hits must rewrite source, cache, and index addresses and each
core's logical update number; retaining only addresses is incorrect when the
cached program's core count or row assignment changes. The host simulator
`../validate_direct_kv_layout.py` is byte-exact for `M=1,2,14,15,29,32`, and
the actual Wormhole BRISC kernel compiles against pinned tt-metal 13adda80.
`[metalium-kv-direct-write] calls=... rows=...` is the qualification proof.
Use `--require-direct-kv-write` together with the ordinary Metalium validator
gates; it requires complete Qwen coverage, not merely a nonzero counter. For
the complete promoted run the exact totals are 1,440 calls and 19,368 rows.
The 17-call/250-row verification core contributes 1,224/18,000; two 2-row
warmups and the 15-row prompt pass contribute the remaining 216/1,368. The
gate derives this from twice the complete batched-QKV telemetry. Screen this
path separately from output-store elision so any gain or failure has one cause.

In the preserved 26-call inspector trace, the five old KV stages appear 1,872
times each: workloads 95-99 cover 25 calls and workloads 144-148 cover the
first call. Direct KV turns 9,360 workload commits into 1,872, removing 7,488
dispatches, about 13% of that trace's decode-window commits. Treat this as a
launch-count bound only; device timing still decides whether the fusion wins.

The previous two-branch autoregressive tree remains a useful control. Its
best depth used 19 calls over 214 rows and measured 41.568 tok/s warm mean.
At matched depth eight, the one-tile batched-QKV merge improved the mean by
about 2.1%. Tree timing put target decode near 2.00 seconds and target KV
preparation below 1 ms, reinforcing that call count dominates logical row
count while the batch stays inside one physical tile.

Precision changes must be judged end to end. Global no-FP32 math preserved an
exact target prefix but changed draft rankings: acceptance fell from 126/233
to 120/263, target calls rose from 17 to 20, and throughput fell to 45.572
tok/s. Output-only HiFi2 and HiFi4/no-FP32 variants were byte-inexact. Faster
isolated matmul is not useful if it changes target tokens or draft acceptance.

## Zero-copy paged attention

A canonical tiled KV cache `[S,H*D]` has the same physical tile ordering as a
32-token paged cache `[S/32,H,32,D]`. The canonical tile index
`seq_tile * H * D_tiles + head * D_tiles + col_tile` is exactly the paged tile
index under an identity page table, and the row/column location within each
tile is unchanged. Use the member `Tensor::reshape()` for the zero-copy view;
`ttnn::reshape()` is an executing operation and defeats the optimization.
Chunked SDPA also bounds its working K length to the aligned current prefix,
avoiding regular attention over the full padded cache and its dense mask.

`GGML_METALIUM_KV_PAGED_ATTENTION=1` is a built but device-unqualified DFlash
experiment. With tracing disabled, batch one, and `2 <= Sq <= 15`, it requires
equal consecutive K/V cache indices, an exact prefix-causal host mask, and
canonical interleaved-DRAM BF16 cache storage. A custom tiled data-movement
operation zero-fills a 32- or 64-row Q tensor and copies each real row to the
front offset aligned to the absolute token position. It then calls chunked SDPA
on the zero-copy paged K/V views and slices out only the real query rows. The
leading Q offset is essential: it makes chunked SDPA's internal causal row
number equal the original token position.

Generic `ttnn::pad` cannot implement that alignment on this runtime because
device padding supports only the end of dimensions. `paged_q_align_dm` instead
copies the two physical 32-byte face-row halves and zeroes every output tile.
Its host byte model covers every offset and length 1 through 15, and the actual
Wormhole BRISC source compiles offline against the preserved runtime.

Host I32-vector and mask shadows are captured and copied only while this gate
is enabled, avoiding baseline overhead. Identity page tables are cached by
active KV length and released before backend runtime teardown. The host proof
`../validate_paged_attention_layout.py` checks physical TT BF16 byte locations,
Q-align zero-fill and face-row placement for every query offset and length 1
through 15, output slicing, causal equivalence, and malformed-mask rejection.

Qualification telemetry is
`[metalium-kv-paged-attention] enabled=1 calls=... rows=... q32=... q64=...`.
Use `--require-paged-attention` with the normal Metalium DFlash gates. The
retained run must match batched QKV exactly: 720 calls and 9,684 rows, with
`q32 + q64 = 720`. Of that, 612/9,000 is the 17-call/250-row verification core;
two 2-row warmups and the 15-row prompt pass account for the remainder. Screen
it separately from direct KV and output-copy elision; otherwise a performance
or correctness change cannot be attributed.

Pair every promoted-path candidate gate with `--require-promoted-dflash`.
That preset pins depth 14, 143 outputs, 233 drafted/126 accepted tokens, 17
target calls, 250 verifier rows, and the full 720-call/9,684-row QKV totals.

## Fused cross-ASIC FFN SwiGLU

The ordinary gated FFN is two parallel projections followed by split SwiGLU.
When the projections live on different ASICs, keeping their raw results inside
the cross-ASIC executor lets one BinaryNg multiply apply `SILU` to the gate
operand. This removes the standalone activation and both intermediate stores;
only the final SwiGLU tensor is materialized.

In the retained 26-call inspector trace, the standalone unary SFPU and its
following binary SFPU each occur 936 times. Fusion therefore removes one
device workload per layer/call, about 1.63% of the 57,336 decode-window commits;
any allocation or output-store reduction is additional.

Treat this as a graph rewrite, not a name-only optimization. Require the exact
gate/up source order, SwiGLU with no swapped input, one use of each projection,
no output flags, and no tracing. Shared, biased, scaled, clamped, or otherwise
noncanonical forms must retain the ordinary path. The default-off gate is
`GGML_METALIUM_INTERNAL_FFN_SWIGLU=1`; qualification telemetry must cover every
batched-QKV layer call (720 calls and 9,684 rows for promoted depth-14 DFlash).

## Model-portability lessons

Qwen3-8B is now the portability control. It has no Glimmer attention-gate projection, yet target-only peer FFN overlap plus DFlash and tree verification work exactly. This confirms that the useful path is not Glimmer-specific.

Draft quantization quality is non-monotonic for speculation. Q4_K_M outperformed the aligned Q8 draft because its candidate ranking covered the greedy target better. Faster community Q4_0 and Q2_K CPU drafts lost end-to-end because acceptance deteriorated. Always gate quant changes on exact target output and full speculative throughput, not draft tokens per second alone.

Future portability work should test another 7B/8B architecture and different hidden/intermediate widths, then derive peer placement from the actual layer split rather than hard-coded Glimmer names.

## Common tt-metal traps

- Many configuration checks are cached in function-local `static` variables;
  set environment variables before process start and use a fresh process for
  every mode.
- `distribute_tensor` accepts host tensors, not device tensors.
- TT mesh metadata in this pinned runtime does not reliably reveal mapper
  sharding; the backend tracks it explicitly with `m3_sharded`.
- A working fabric configuration must be selected before device open.
- JIT warmth is part of the steady-state gate: an exact DFlash run with seven misses fell from the 54.791 tok/s warm mean to 35.269 tok/s. JIT hits still do not remove model upload or trace-capture time; benchmark decode separately from CLI startup.
- Zero host fallback and token equality are gates, not optional diagnostics.
- Greedy target equality alone is insufficient for speculation. Also compare acceptance, target-call count, and verifier rows; global no-FP32 math kept the prefix exact but lost about 17% throughput through worse draft rankings.
- Keep `QWEN_TREE_TEMP=0` for a greedy target. `LLAMA_SPEC_TREE_DRAFT_TEMP` is the separate draft-only control.
- For an N-branch tree, keep draft top-k at least N. The retained two-branch mode uses top-k 2; lower values are unsafe for the current tree loop.
- Faster isolated draft decode does not imply faster speculation. Measure acceptance, target batch size, and end-to-end output rate.
- Fewer verifier rows do not necessarily mean lower TT latency. Target call count and accepted output per call dominate in the tested depth-5-to-11 range.
- Floating tensors are tiled in 32-row units. A logical M between 2 and 32 still occupies one M tile; optimize work inside that paid tile.
- The batched-QKV merge is verified in both the promoted DFlash runner and `qwen3-tree-tt-best`. Keep it opt-in for other modes until their shapes pass the same byte-exact, path-use, fabric, and teardown gates.
- Repeated ETH-heartbeat startup failures are a runtime/firmware reliability problem, not a model result. Preserve the exact failure log and stop; do not enter a reset/retry loop.

## DSpark / DFlash block speculation

The official Qwen3-8B DSpark sidecar is a `dflash` GGUF, not a `dspark` GGUF. `ggml-org/Qwen3-8B-GGUF/dspark-Qwen3-8B-Q8_0.gguf` loads in this checkout and has SHA256 `441880f9a7f10021c2f50702d845702ce39fab95753d2264f27011f32e649252`. A community Q4 file declaring architecture `dspark` fails cleanly as unsupported and is not a usable quantization control.

With a seven-token block and a greedy Qwen3-8B Q8 target, exact off-device means are 298.321 tok/s for CUDA/CUDA, 177.688 tok/s for a CUDA target plus CPU sidecar after warm-up, and 85.571 tok/s for CPU/CPU. The common CUDA-placement runs accept 112 of 147 drafted tokens, need 21 target calls over 168 verifier rows, and emit 6.333 output tokens per target call. This is substantially better target-call efficiency than the autoregressive tree drafter.

Depth seven is the Metalium candidate even though depth six is tied on the CUDA/CPU control. Seven uses one fewer target call, and verifier sizes two through eight all pay the same single 32-row M tile on TT.

The DSpark context borrows target token embeddings and output projection through `ctx_other`. A nominal CPU draft can therefore retain accelerator scheduler splits for those shared tensors; the CUDA/CPU control reported 68 splits. If Metalium cannot place or schedule the shared tensors, add the target model's backends to the draft context as in upstream PR 26636 before altering speculation semantics.

Always feed target features through `common_speculative_process`. Upstream issue 26884 shows that omitting this input can silently collapse DFlash/DSpark acceptance while still producing apparently valid output. Exact-prefix and counter validation are mandatory.

The old 49-51 tok/s DSpark N300 estimate was never executed and is now below
the verified 54.791 tok/s DFlash result. DSpark needs 21 target calls rather
than 17 and is deprioritized without a new scheduling or model hypothesis.

## DFlash block-16 result

The authoritative llama.cpp checkpoint is `ggml-org/Qwen3-8B-GGUF/dflash-Qwen3-8B-Q8_0.gguf`, SHA256 `89a3d462b8e945c602599a91ab2329aa387f677d88383d0e23253b9f0082e502`. It is a five-layer `dflash` sidecar with block size 16. Ordinary DFlash uses the first position as the seed, so its maximum draft is 15 tokens.

Depth 14 is a sharp local optimum on the fixed quicksort prompt. It emits 143 output tokens, accepts 126 of 233 drafted tokens, needs 17 target calls over 250 verifier rows, and reaches 8.412 output tokens per target call. Depth 13 needs 19 calls over 262 rows; depth 15 needs 18 calls over 281 rows. Since all depth-14 verification batches have at most 15 rows, they fit one Metalium M tile.

Exact off-device means are 343.059 tok/s for CUDA/CUDA and 226.566 tok/s for CUDA target plus CPU draft. The instrumented CPU/CPU depth-14 control is 83.554 tok/s. DFlash is 15% faster than DSpark on all CUDA and roughly 27.5% faster on the hybrid placement control, despite a lower raw acceptance percentage.

The hybrid timing attributes 328.783 ms to 17 DFlash calls and 289.13 ms to target evaluation. The CPU timing attributes 342.977 ms to drafting and 1248.98 ms to target evaluation. Block drafting is cheap enough that target call count remains the primary Metalium objective.

Verified N300 ordinary DFlash measured 54.509 and 55.072 tok/s warm, a
54.791 tok/s mean. Both runs emitted 143 exact output tokens, accepted 126/233
drafts, needed 17 target calls over 250 verifier rows, fused 720/720 QKV
groups, used 1438/0 direct/fallback fabric transfers, hit 301/301 JIT entries,
and shut down cleanly. An exact same-shape run with seven JIT misses measured
only 35.269 tok/s.

DFlash generation took 642.598 and 620.848 ms and sampling took 99.97 and
93.00 ms. Roughly 1.88 seconds remained in target decode plus loop overhead.
Full TT draft offload did not reduce draft time and measured 45.521 tok/s cold.
A local Q8 output head was exact but slower at a 53.479 tok/s mean. Global
no-FP32 math stayed exact but reduced acceptance and fell to 45.572 tok/s.
The alternate-tree/full-block path and output-only HiFi2/HiFi4 paths were
byte-inexact.

The isolated physical LM-head workload `[32,4096] x [4096,151936]` is about
4 ms per target call. A 32-core, eight-chunk DRAM-sharded HiFi2 probe reached
2.867 ms including concat, but HiFi2 end-to-end output was invalid. The
backend's default-off `GGML_METALIUM_LM_HEAD_FAST_COMPUTE=1` gate uses
output-only HiFi3/no-FP32 and remains unqualified after heartbeat failures.
Test it exactly once on a future healthy device; remove it if prefix,
acceptance, or throughput regresses. Even an ideal result saves only tens of
milliseconds per generation, so decoder-layer dispatch remains the main
lever.

For a CPU draft with shared target tensors, keep the target device explicitly in the draft device list even with `n_gpu_layers_draft=0`. That makes the target-owned embedding and output projection schedulable without PR 26636. The upstream patch is needed only when the draft device list intentionally excludes devices that own shared target tensors.

Trace-level `sched_reserve` messages can interleave with generated text. Correctness tooling may strip only lines matching the exact timestamped `sched_reserve:` diagnostic prefix; do not remove arbitrary timestamped output.

## Rectangular target tracing

Expanding the depth-10 two-branch verifier into a fixed 2 x 11 rectangle made one target graph structurally reusable. A telemetry run observed four graph states, captured one, and replayed it 17 times. It still measured only 40.970 tok/s; two neighboring exact runs measured 41.412 and 41.595 tok/s. Target decode stayed around 2.01-2.03 seconds versus 1.999 seconds for the compact control.

The rectangle paid 418 target rows over 19 calls, including 156 padding rows, but remained inside one M tile per call. The neutral result proves that eliminating the host graph walk and replaying the TT command stream is not enough; the decoder programs and synchronization inside each target call dominate.

Trace capture is memory-tight. The telemetry run reported 511.3 MiB allocated, 509.8 MiB free, 9.5 MiB of pinned intermediates, and 7709.5 MiB of pinned I/O. A compact structural-key variant subsequently wedged mid-generation. Keep `GGML_METALIUM_TRACE_STRUCTURAL_KEY` disabled unless a new alias/lifetime proof is implemented and tested.

`[trace-mem captured]` is emitted on stdout and can land inside generated text. Validation tooling must remove only that exact diagnostic before byte comparison; broad log-line stripping can hide a real correctness failure.
