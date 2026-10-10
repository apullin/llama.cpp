#include "paged_q_align.hpp"

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

namespace paged_q_align_device {

PagedQAlignDeviceOperation::program_factory_t
PagedQAlignDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*t*/) {
    return program::PagedQAlignProgramFactory{};
}

void PagedQAlignDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& a, const tensor_args_t& t) {
    TT_FATAL(t.src.storage_type() == ttnn::StorageType::DEVICE,
        "paged-q-align: src must be on device");
    TT_FATAL(t.src.layout() == tt::tt_metal::Layout::TILE,
        "paged-q-align: src must be TILE");
    TT_FATAL(t.src.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "paged-q-align: src must be BF16");
    TT_FATAL(t.src.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED,
        "paged-q-align: src must be interleaved");
    TT_FATAL(t.src.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM,
        "paged-q-align: src must be in DRAM");

    const auto src = t.src.logical_shape();
    TT_FATAL(src.rank() == 4 && src[0] == 1,
        "paged-q-align: expected [1,H,Sq,D], got {}", src);
    TT_FATAL(a.rows > 0 && a.rows <= 32 && src[2] == a.rows,
        "paged-q-align: source rows must be in [1,32]");
    TT_FATAL(a.heads > 0 && src[1] == a.heads,
        "paged-q-align: head count must match src");
    TT_FATAL(a.head_dim >= 32 && a.head_dim <= 1024 && a.head_dim % 32 == 0 && src[3] == a.head_dim,
        "paged-q-align: head dimension must be a multiple of 32 and at most 1024");
    TT_FATAL(a.aligned_rows == 32 || a.aligned_rows == 64,
        "paged-q-align: output must have 32 or 64 rows");
    TT_FATAL(a.q_offset < 32 && a.q_offset + a.rows <= a.aligned_rows,
        "paged-q-align: real rows do not fit the aligned output");
    TT_FATAL(t.src.padded_shape()[2] == 32 && t.src.padded_shape()[3] == a.head_dim,
        "paged-q-align: source must occupy exactly one padded query tile row");
    TT_FATAL(t.src.buffer()->page_size() == sizeof(bfloat16) * 32 * 32,
        "paged-q-align: source page must be one BF16 tile");
    TT_FATAL(a.heads <= t.src.device()->compute_with_storage_grid_size().x *
        t.src.device()->compute_with_storage_grid_size().y,
        "paged-q-align: not enough worker cores for one core per head");
}

void PagedQAlignDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& a, const tensor_args_t& t) {
    validate_on_program_cache_miss(a, t);
}

PagedQAlignDeviceOperation::spec_return_value_t
PagedQAlignDeviceOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t& t) {
    return TensorSpec(
        ttnn::Shape({1, a.heads, a.aligned_rows, a.head_dim}),
        tt::tt_metal::TensorLayout(
            t.src.dtype(),
            tt::tt_metal::PageConfig(t.src.layout()),
            t.src.memory_config()));
}

PagedQAlignDeviceOperation::tensor_return_value_t
PagedQAlignDeviceOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& t) {
    return ttnn::create_device_tensor(compute_output_specs(a, t), t.src.device());
}

std::tuple<operation_attributes_t, tensor_args_t>
PagedQAlignDeviceOperation::invoke(
    const Tensor& src, uint32_t q_offset, uint32_t aligned_rows) {
    const auto shape = src.logical_shape();
    return {
        operation_attributes_t{
            q_offset,
            aligned_rows,
            (uint32_t)shape[2],
            (uint32_t)shape[1],
            (uint32_t)shape[3]},
        tensor_args_t{src}};
}

namespace program {

PagedQAlignProgramFactory::cached_program_t PagedQAlignProgramFactory::create(
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    Program prog = CreateProgram();
    using CB = tt::CBIndex;

    IDevice* device = tensor_args.src.device();
    const auto grid = device->compute_with_storage_grid_size();
    const CoreRangeSet all_cores = num_cores_to_corerangeset(attrs.heads, grid, true);
    const auto cores = corerange_to_cores(all_cores, attrs.heads, true);

    const uint32_t tile_bytes = sizeof(bfloat16) * 32 * 32;
    CreateCircularBuffer(
        prog, all_cores,
        CircularBufferConfig(tile_bytes, {{CB::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(CB::c_0, tile_bytes));

    std::vector<uint32_t> ct{attrs.head_dim / 32};
    TensorAccessorArgs(*tensor_args.src.buffer()).append_to(ct);
    TensorAccessorArgs(*tensor_return_value.buffer()).append_to(ct);
    const KernelHandle dm = CreateMetaliumKernel(
        prog, "paged_q_align_dm", all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct});

    for(uint32_t head = 0; head < attrs.heads; ++head) {
        SetRuntimeArgs(prog, dm, cores[head], std::vector<uint32_t>{
            (uint32_t)tensor_args.src.buffer()->address(),
            (uint32_t)tensor_return_value.buffer()->address(),
            head,
            attrs.rows,
            attrs.q_offset,
            attrs.aligned_rows / 32});
    }

    return {std::move(prog), shared_variables_t{dm, cores}};
}

void PagedQAlignProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attrs,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    const auto dm = cached_program.shared_variables.dm;
    const auto& cores = cached_program.shared_variables.cores;
    for(uint32_t head = 0; head < cores.size(); ++head) {
        auto& ra = GetRuntimeArgs(program, dm, cores[head]);
        ra[0] = (uint32_t)tensor_args.src.buffer()->address();
        ra[1] = (uint32_t)tensor_return_value.buffer()->address();
        ra[2] = head;
        ra[3] = attrs.rows;
        ra[4] = attrs.q_offset;
        ra[5] = attrs.aligned_rows / 32;
    }
}

} // namespace program

} // namespace paged_q_align_device

ttnn::Tensor paged_q_align(const Tensor& src, uint32_t q_offset, uint32_t aligned_rows) {
    auto [attrs, args] =
        paged_q_align_device::PagedQAlignDeviceOperation::invoke(src, q_offset, aligned_rows);
    return ttnn::device_operation::detail::launch<paged_q_align_device::PagedQAlignDeviceOperation>(attrs, args);
}

} // namespace ttggml
