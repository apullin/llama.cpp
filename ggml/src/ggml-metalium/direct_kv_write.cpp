#include "direct_kv_write.hpp"

#include <array>
#include <vector>

#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "tt-metalium/tt_backend_api_types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <ttnn/tensor/layout/layout.hpp>
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "utils.hpp"

using namespace tt::tt_metal;

namespace ttggml {
using namespace ttnn;

namespace direct_kv_device {

DirectKvWriteDeviceOperation::program_factory_t
DirectKvWriteDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*t*/) {
    return program::DirectKvProgramFactory{};
}

void DirectKvWriteDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    TT_FATAL(t.src.storage_type() == tt::tt_metal::StorageType::DEVICE, "direct-kv: src must be on device");
    TT_FATAL(t.cache.storage_type() == tt::tt_metal::StorageType::DEVICE, "direct-kv: cache must be on device");
    TT_FATAL(t.indices.storage_type() == tt::tt_metal::StorageType::DEVICE, "direct-kv: indices must be on device");
    TT_FATAL(t.src.device() == t.cache.device() && t.src.device() == t.indices.device(),
        "direct-kv: all tensors must be on the same device");
    TT_FATAL(t.src.layout() == tt::tt_metal::Layout::TILE, "direct-kv: src must be TILE");
    TT_FATAL(t.cache.layout() == tt::tt_metal::Layout::TILE, "direct-kv: cache must be TILE");
    TT_FATAL(t.indices.layout() == tt::tt_metal::Layout::ROW_MAJOR, "direct-kv: indices must be ROW_MAJOR");
    TT_FATAL(t.src.dtype() == tt::tt_metal::DataType::BFLOAT16, "direct-kv: src must be BF16");
    TT_FATAL(t.cache.dtype() == tt::tt_metal::DataType::BFLOAT16, "direct-kv: cache must be BF16");
    TT_FATAL(t.indices.dtype() == tt::tt_metal::DataType::UINT32, "direct-kv: indices must be UINT32");
    TT_FATAL(t.src.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "direct-kv: src must be interleaved");
    TT_FATAL(t.cache.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "direct-kv: cache must be interleaved");
    TT_FATAL(t.indices.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "direct-kv: indices must be interleaved");
    TT_FATAL(t.src.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
        "direct-kv: src must be in DRAM");
    TT_FATAL(t.cache.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
        "direct-kv: cache must be in DRAM");
    TT_FATAL(t.indices.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
        "direct-kv: indices must be in DRAM");

    const auto src = t.src.logical_shape();
    const auto cache = t.cache.logical_shape();
    TT_FATAL(src.rank() == 4 && cache.rank() == 4, "direct-kv: tensors must be rank 4");
    TT_FATAL(src[0] == 1 && src[1] == 1 && cache[0] == 1 && cache[1] == 1,
        "direct-kv: expected [1,1,M,W] and [1,1,S,W]");
    TT_FATAL(a.updates > 0 && a.updates <= 32 && src[2] == a.updates,
        "direct-kv: updates must match src rows and fit one tile");
    TT_FATAL(a.width == 1024 && src[3] == a.width && cache[3] == a.width,
        "direct-kv: only the Qwen width-1024 path is enabled");
    TT_FATAL(cache[2] == a.cache_rows && a.cache_rows % 32 == 0,
        "direct-kv: cache rows must be tile aligned");
    TT_FATAL(t.src.padded_shape()[2] == 32 && t.src.padded_shape()[3] == a.width,
        "direct-kv: source must occupy exactly one padded row tile");
    TT_FATAL(t.indices.logical_shape().volume() == a.updates,
        "direct-kv: index count must match updates");
    TT_FATAL(t.src.buffer()->page_size() == sizeof(bfloat16) * 32 * 32,
        "direct-kv: source page must be one BF16 tile");
    TT_FATAL(t.cache.buffer()->page_size() == sizeof(bfloat16) * 32 * 32,
        "direct-kv: cache page must be one BF16 tile");
    TT_FATAL(t.indices.buffer()->num_pages() == 1,
        "direct-kv: indices must occupy one row-major stick");
    TT_FATAL(t.indices.buffer()->page_size() >= a.updates * sizeof(uint32_t),
        "direct-kv: index stick is too small");
    TT_FATAL(a.updates <= t.src.device()->compute_with_storage_grid_size().x *
        t.src.device()->compute_with_storage_grid_size().y,
        "direct-kv: not enough worker cores");
}

void DirectKvWriteDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& a, const tensor_args_t& t) {
    validate_on_program_cache_miss(a, t);
}

DirectKvWriteDeviceOperation::spec_return_value_t
DirectKvWriteDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*a*/, const tensor_args_t& t) {
    return t.cache.tensor_spec();
}

DirectKvWriteDeviceOperation::tensor_return_value_t
DirectKvWriteDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*a*/, const tensor_args_t& t) {
    return t.cache;
}

std::tuple<operation_attributes_t, tensor_args_t>
DirectKvWriteDeviceOperation::invoke(
    const Tensor& src, const Tensor& cache, const Tensor& indices) {
    const auto ss = src.logical_shape();
    const auto cs = cache.logical_shape();
    return {
        operation_attributes_t{(uint32_t)ss[2], (uint32_t)ss[3], (uint32_t)cs[2]},
        tensor_args_t{src, cache, indices}};
}

namespace program {

DirectKvProgramFactory::cached_program_t DirectKvProgramFactory::create(
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    Program prog = CreateProgram();
    using CB = tt::CBIndex;

    IDevice* device = tensor_args.src.device();
    const auto grid = device->compute_with_storage_grid_size();
    const CoreRangeSet all_cores = num_cores_to_corerangeset(attrs.updates, grid, true);
    const auto cores = corerange_to_cores(all_cores, attrs.updates, true);

    const uint32_t tile_bytes = sizeof(bfloat16) * 32 * 32;
    const uint32_t row_bytes = attrs.width * sizeof(bfloat16);
    const uint32_t row_tiles = (row_bytes + tile_bytes - 1) / tile_bytes;
    CreateCircularBuffer(
        prog, all_cores,
        CircularBufferConfig(row_tiles * tile_bytes, {{CB::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(CB::c_0, tile_bytes));
    CreateCircularBuffer(
        prog, all_cores,
        CircularBufferConfig(32, {{CB::c_1, tt::DataFormat::UInt32}})
            .set_page_size(CB::c_1, 32));

    std::vector<uint32_t> ct{attrs.width / 32, attrs.cache_rows};
    TensorAccessorArgs(*tensor_args.src.buffer()).append_to(ct);
    TensorAccessorArgs(*tensor_return_value.buffer()).append_to(ct);
    TensorAccessorArgs(*tensor_args.indices.buffer()).append_to(ct);
    const KernelHandle dm = CreateMetaliumKernel(
        prog, "direct_kv_write_dm", all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct});

    for(uint32_t update = 0; update < attrs.updates; ++update) {
        SetRuntimeArgs(prog, dm, cores[update], std::vector<uint32_t>{
            (uint32_t)tensor_args.src.buffer()->address(),
            (uint32_t)tensor_return_value.buffer()->address(),
            (uint32_t)tensor_args.indices.buffer()->address(),
            update});
    }

    return {std::move(prog), shared_variables_t{dm, cores}};
}

void DirectKvProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    const auto dm = cached_program.shared_variables.dm;
    const auto& cores = cached_program.shared_variables.cores;
    for(uint32_t update = 0; update < cores.size(); ++update) {
        auto& ra = GetRuntimeArgs(program, dm, cores[update]);
        ra[0] = (uint32_t)tensor_args.src.buffer()->address();
        ra[1] = (uint32_t)tensor_return_value.buffer()->address();
        ra[2] = (uint32_t)tensor_args.indices.buffer()->address();
        ra[3] = update;
    }
}

} // namespace program

} // namespace direct_kv_device

ttnn::Tensor direct_kv_write(
    const Tensor& src, const Tensor& cache, const Tensor& indices) {
    auto [attrs, args] = direct_kv_device::DirectKvWriteDeviceOperation::invoke(src, cache, indices);
    return ttnn::device_operation::detail::launch<direct_kv_device::DirectKvWriteDeviceOperation>(attrs, args);
}

} // namespace ttggml
