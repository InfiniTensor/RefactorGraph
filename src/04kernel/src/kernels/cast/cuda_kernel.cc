#include "cuda_kernel.hh"

#ifdef USE_CUDA
#include "../../generator/nvrtc_repo.h"
#include "kernel/cuda/threads_distributer.cuh"
#include <cuda_runtime.h>
#endif

namespace refactor::kernel {
    using K = CastCuda;

    K::CastCuda(decltype(from) from_,
                decltype(to) to_,
                decltype(size) size_) noexcept
        : from(from_), to(to_), size(size_) {}

    auto K::build(Tensor const &from, Tensor const &to) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return std::make_unique<K>(from.dataType, to.dataType, from.elementsSize());
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing cast operation on Nvidia gpu";
    }

#ifdef USE_CUDA

    constexpr static const char *TEMPLATE = R"~(
extern "C" __global__ void kernel(
    {1:}       *__restrict__ y,
    {0:} const *__restrict__ x,
    size_t n
) {{
    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < n;
         tid += step)
        y[tid] = {2:};
}}
)~";

    constexpr static uint16_t __(DataType from, DataType to) {
        return (static_cast<uint16_t>(from) << 8) | static_cast<uint16_t>(to);
    }

    static std::string cast(DataType from, DataType to) {
        using DT = DataType;
        if (from.isCpuNumberic()) {
            if (to.isCpuNumberic()) {
                return fmt::format("x[tid]", nvrtc::dataType(to));
            }
            if (!from.isFloat()) {
                return to == DT::FP16
                           ? "__int2half_rz(x[tid])"
                           : "__int2bfloat16_rz(x[tid])";
            }
        } else if (to.isCpuNumberic()) {
            if (!to.isFloat()) {
                return from == DT::FP16
                           ? "__half2int_rz(x[tid])"
                           : "__bfloat162int_rz(x[tid])";
            }
        }
        // clang-format off
        static const std::unordered_map<uint16_t, std::string> OP{
            {__(DT::FP16, DT::BF16), "__float2bfloat16(__half2float(x[tid]))" },
            {__(DT::FP16, DT::F32 ), "__half2float(x[tid])"                   },
            {__(DT::FP16, DT::F64 ), "__half2float(x[tid])"                   },
            {__(DT::FP16, DT::Bool), "x[tid] == CUDART_ZERO_FP16"             },

            {__(DT::BF16, DT::FP16), "__float2half(__bfloat162float(x[tid]))" },
            {__(DT::BF16, DT::F32 ), "__bfloat162float(x[tid])"               },
            {__(DT::BF16, DT::F64 ), "__bfloat162float(x[tid])"               },
            {__(DT::BF16, DT::Bool), "x[tid] == CUDART_ZERO_BF16"             },

            {__(DT::F32 , DT::FP16), "__float2half(x[tid])"     },
            {__(DT::F32 , DT::BF16), "__float2bfloat16(x[tid])" },
            {__(DT::F32 , DT::Bool), "x[tid] == .0f"            },

            {__(DT::F64 , DT::FP16), "__float2half(x[tid])"     },
            {__(DT::F64 , DT::BF16), "__float2bfloat16(x[tid])" },
            {__(DT::F64 , DT::Bool), "x[tid] == .0"             },
        };
        // clang-format on
        return OP.at(__(from, to));
    }

    auto K::lower(Resources &res) const -> RoutineWorkspace {
        using namespace runtime;

        if (from == to) {
            return [n = size * from.size()](Resources &, void *, void const *const *inputs, void *const *outputs) {
                if (outputs[0] != inputs[0]) {
                    cudaMemcpyAsync(outputs[0], inputs[0], n, cudaMemcpyDeviceToDevice);
                }
            };
        }

        auto name = fmt::format("cast_{}->{}", nvrtc::dataType(from), nvrtc::dataType(to));
        auto code = fmt::format(TEMPLATE, nvrtc::dataType(from), nvrtc::dataType(to), cast(from, to));
        return [h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "kernel"),
                params = cuda::ThreadsDistributer()(size)](
                   Resources &, void *, void const *const *inputs, void *const *outputs) {
            auto y = outputs[0];
            auto x = inputs[0];
            auto n = params.n;
            void *args[]{&y, &x, &n};
            CUDA_ASSERT(cuLaunchKernel(
                h->kernel(),
                params.gridSize, 1, 1,
                params.blockSize, 1, 1,
                0, nullptr, args, nullptr));
        };
    }

#endif

}// namespace refactor::kernel
