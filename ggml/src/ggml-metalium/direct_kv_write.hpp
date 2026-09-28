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

namespace direct_kv_device {

struct operation_attributes_t {
    uint32_t updates;
    uint32_t width;
    uint32_t cache_rows;
};

struct tensor_args_t {
    const Tensor& src;
    const Tensor& cache;
    const Tensor& indices;
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {

struct DirectKvProgramFactory {
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

struct DirectKvWriteDeviceOperation {
    using operation_attributes_t = direct_kv_device::operation_attributes_t;
    using tensor_args_t = direct_kv_device::tensor_args_t;
    using spec_return_value_t = direct_kv_device::spec_return_value_t;
    using tensor_return_value_t = direct_kv_device::tensor_return_value_t;
    using program_factory_t = std::variant<program::DirectKvProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& src, const Tensor& cache, const Tensor& indices);
};

} // namespace direct_kv_device

// Directly scatter the real rows of tiled BF16 src [1,1,M,1024] into tiled BF16
// cache [1,1,S,1024] using UINT32 row indices. In-place; returns the cache tensor.
ttnn::Tensor direct_kv_write(
    const Tensor& src, const Tensor& cache, const Tensor& indices);

} // namespace ttggml
