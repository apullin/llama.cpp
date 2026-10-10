#pragma once

#include <tuple>
#include <variant>
#include <vector>

#include <ttnn/device_operation.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>

namespace ttggml {
using namespace ttnn;

namespace paged_q_align_device {

struct operation_attributes_t {
    uint32_t q_offset;
    uint32_t aligned_rows;
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
};

struct tensor_args_t {
    const Tensor& src;
};

using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {

struct PagedQAlignProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle dm;
        std::vector<tt::tt_metal::CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&);

    static void override_runtime_arguments(
        cached_program_t&,
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&);
};

} // namespace program

struct PagedQAlignDeviceOperation {
    using operation_attributes_t = paged_q_align_device::operation_attributes_t;
    using tensor_args_t = paged_q_align_device::tensor_args_t;
    using spec_return_value_t = paged_q_align_device::spec_return_value_t;
    using tensor_return_value_t = paged_q_align_device::tensor_return_value_t;
    using program_factory_t = std::variant<program::PagedQAlignProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& src, uint32_t q_offset, uint32_t aligned_rows);
};

} // namespace paged_q_align_device

// Align tiled BF16 Q [1,H,Sq,D] to an absolute 32-row chunk by zero-filling
// [1,H,32|64,D] and copying each real row to q_offset + row.
ttnn::Tensor paged_q_align(const Tensor& src, uint32_t q_offset, uint32_t aligned_rows);

} // namespace ttggml
