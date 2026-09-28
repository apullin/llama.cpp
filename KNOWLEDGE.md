# Tenstorrent N300 Dual-ASIC Knowledge Log

## Environment
- TT_METAL_RUNTIME_ROOT=/home/pullin/personal/tenstorrent/generated/tt-metal-13adda80 required for runtime.
- TT_METAL_HOME same path for build.
- Mesh activation: GGML_METALIUM_MESH_SHAPE=1x2
- Device open: ttnn::distributed::open_mesh_device(mesh_shape, ...)
- Mesh num devices recorded in dev_ctx.mesh_num_devices

## Build
- Build with TT_METAL_HOME set, else CMake fails.
- ggml-metalium.cpp changes require rebuild of ggml-metalium target.

## Distributed APIs (tt-metal 13adda80)
- shard_tensor_to_mesh_mapper signature:
  std::unique_ptr<TensorToMesh> ttnn::distributed::shard_tensor_to_mesh_mapper(MeshDevice&, int dim, std::optional<int> cluster_axis)
  File: build_Release/include/ttnn/distributed/distributed_tensor.hpp:73
- distribute_tensor signature:
  Tensor ttnn::distributed::distribute_tensor(const Tensor& tensor, const TensorToMesh& mapper, std::optional<std::reference_wrapper<MeshDevice>> mesh_device)
  File: build_Release/include/ttnn/distributed/distributed_tensor.hpp:113
  Note: TensorToMesh passed by const ref, not unique_ptr. Requires *mapper.
  Note: distribute_tensor only supports host tensors; device tensor causes TT_FATAL.
- all_gather:
  ttnn::all_gather(out, -1, cluster_axis)
  Header: ttnn/operations/ccl/all_gather/all_gather.hpp

## Sharding Policy M3
- Shard 2D weight tensors with ne1 %64==0, size >=1MiB, non_one_dims==2
- Shard along TT dim 2 (corresponds to GGML ne1)
- 1D norms/biases replicated
- KV cache replicated
- buffer_set_tensor replacement: to_device -> distribute_tensor via mapper
- all_gather after matmul when weight sharded

## Gotchas
- distribute_tensor expects a host tensor, not a device tensor. Distribute before to_device, then tilize.
- A sharded weight and `[K,N]` storage can intentionally have a non-canonical physical shape. Only those storage-boundary checks are relaxed; computed output and buffer_get_tensor shape assertions stay enabled.
- Use `ggml_tensor_extra_metalium::m3_sharded` on the GGML weight tensor (`src0`) as the gather predicate. Do not infer sharding from the activation or a transient TTNN layout.
- The original `[202048,1,1,1]` versus `[101024,1,1,1]` mismatch was a missing post-matmul gather, not a reason to disable output shape assertions.

## Hardware
- N300 = 2 Wormhole ASICs, 12GB each, 24GB total
- Single-ASIC baseline: -ngl20 --no-kv-offload
- -ngl99 OOMs single ASIC, requires mesh
- KV offload not supported: always --no-kv-offload

## Correctness Gate
- llama-cli --temp 0 fixed prompt must match token-for-token single-ASIC
- llama-bench -ngl99 must fit and run on 1x2 mesh

## 2026-08-12 Mesh shape mismatch (resolved)
- Symptom was MUL_MAT output width `101024` when GGML required `202048` on the 1x2 mesh.
- Root cause was a missing gather caused by inspecting the wrong tensor for sharding.
- Current code records `m3_sharded` on weight metadata and calls `ggml_metalium_m3_gather(out, src0)`.
- Output and buffer_get_tensor shape assertions are enabled. See `SUPERSEDED.md` for the discarded hypotheses and temporary assertion workarounds.
