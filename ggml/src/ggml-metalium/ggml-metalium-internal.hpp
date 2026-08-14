#pragma once

// Backend internals shared between ggml-metalium.cpp and the graph compiler.
// These are the few types and helpers the lowering dispatch needs to bridge ggml
// tensors and TTNN tensors; everything else stays private to ggml-metalium.cpp.

#include <memory>
#include <optional>
#include <string>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/device.hpp>
#include <ttnn/operations/core/compute_kernel/compute_kernel_config.hpp>

#include "ggml.h"

class MetaliumGraphCompiler;
struct MetaliumBackendRuntime;

// Per ggml-tensor device state, hung off ggml_tensor::extra.
struct ggml_tensor_extra_metalium {
    std::shared_ptr<ttnn::Tensor> tensor;

    // Row-folded physical storage for a tensor whose GGML-declared shape would tile-pad badly
    // (1-D / short-penultimate caches and the token embedding). When non-null, this is the
    // authoritative device copy stored as [n_rows, dim/32, 32] and `tensor` is dropped
    std::shared_ptr<ttnn::Tensor> row_folded;

    std::optional<tt::tt_metal::MemoryConfig> memory_config;

    // M3 dual-ASIC: true when this tensor was sharded across the mesh at upload
    // (column-parallel weight shard along GGML ne1). This tt-metal build reports
    // the per-device shard shape via logical_shape() and MeshBufferLayout::REPLICATED
    // even for mapper-sharded tensors, so distribution must be tracked here rather
    // than via mesh-buffer introspection.
    bool m3_sharded = false;

    // Weight-layout optimization: true when this (matmul-only) weight was stored
    // transposed at upload — [1,1,K,N] instead of GGML's natural [1,1,N,K] — so
    // matmul/linear run with transpose_b=false (~1.4x faster for both bf16 and
    // block-float weights; transpose_b forces strided DRAM reads). The transpose
    // is applied to the row-major tensor before tilize so block-float
    // quantization stays single-pass (a post-tilize transpose dequant+requants).
    bool weight_kn = false;

    bool is_row_folded() const { return row_folded != nullptr; }
};

// True when `w` is a weight stored [K,N] (see ggml_tensor_extra_metalium::weight_kn);
// matmul sites pass transpose_b = !weight_kn(w).
inline bool ggml_metalium_weight_kn(const ggml_tensor * w) {
    auto * m = w != nullptr ? static_cast<ggml_tensor_extra_metalium*>(w->extra) : nullptr;
    return m != nullptr && m->weight_kn;
}

struct ggml_backend_metalium_context {
    ttnn::IDevice* device = nullptr;
    int device_id = 0;
    std::string name;
    std::unique_ptr<MetaliumGraphCompiler> compiler;
    MetaliumBackendRuntime * runtime = nullptr;
};

// Materialises the TTNN tensor backing a (possibly lazily-viewed) ggml tensor.
std::shared_ptr<ttnn::Tensor> realize_ggml_view(const ggml_tensor* tensor);

// Store a normal materialized tensor result while preserving output placement intent.
void ggml_metalium_store_tensor(ggml_tensor_extra_metalium* meta, ttnn::Tensor value);
void ggml_metalium_store_tensor(ggml_tensor_extra_metalium* meta, std::shared_ptr<ttnn::Tensor> value);

// Adapt a TTNN tensor's logical shape to a ggml node's shape (no-op if already equal, else ttnn::reshape).
// Fusions that emit a result in a different-but-equivalent tiling use this so the backend's post-op shape
// sanity check passes and consumers see the node's canonical shape.
ttnn::Tensor reshape_tt_tensor_into_ggml(const ttnn::Tensor& tensor, const struct ggml_tensor * node);

// M3 dual-ASIC: if the weight backing a matmul/linear was sharded across the mesh at
// upload, gather the per-device partial outputs back to full width (host roundtrip;
// see ggml-metalium.cpp for why this is not a CCL all_gather). Otherwise a no-op.
ttnn::Tensor ggml_metalium_m3_gather(ttnn::Tensor out, const ggml_tensor * weight);

// Stable identity for a cgraph: its uid when set, else a topology+shape signature. The graph
// compiler caches its per-graph fusion plan on this key (the same key the trace layer uses).
uint64_t metalium_graph_key(const ggml_cgraph* cgraph);

// The matmul math-fidelity config the backend uses for all GEMMs (HiFi4 on Wormhole). Fusions that
// emit their own matmul/linear must reuse this so accuracy matches the unfused path.
ttnn::DeviceComputeKernelConfig make_compute_kernel_config(ttnn::IDevice* device);

inline void ggml_metalium_op_src_sanity_check(const ggml_tensor * node, int idx) {
    GGML_ASSERT(node->src[idx] != NULL);
    GGML_ASSERT(node->src[idx]->extra != NULL);
    auto* meta = (ggml_tensor_extra_metalium*)(node->src[idx]->extra);
    if(meta->tensor != NULL) {
        GGML_ASSERT(meta->tensor->storage_type() == tt::tt_metal::StorageType::DEVICE);
        GGML_ASSERT(meta->tensor->layout() == tt::tt_metal::Layout::TILE);
    }
}

#define GGML_METALIUM_OP_SANITY_CHECK(_node) \
    GGML_ASSERT((_node)->extra != NULL);
#define GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, _idx) ggml_metalium_op_src_sanity_check(_node, _idx);
#define GGML_METALIUM_OP_SRC0_SANITY_CHECK(_node) GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, 0)
#define GGML_METALIUM_OP_SRC1_SANITY_CHECK(_node) GGML_METALIUM_OP_SRC_SANITY_CHECK(_node, 1)
