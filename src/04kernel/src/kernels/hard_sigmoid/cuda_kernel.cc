#include "cuda_kernel.hh"

#ifdef USE_CUDA
#include "../../generator/nvrtc_repo.h"
#include "kernel/cuda/threads_distributer.cuh"
#include <cuda_runtime.h>
#endif

namespace refactor::kernel {
    using K = HardSigmoidCuda;
    using DT = DataType;

    K::HardSigmoidCuda(float alpha_, float beta_, DT dt_, size_t size_) noexcept
        : Kernel(), alpha(alpha_), beta(beta_), dataType(dt_), size(size_) {}

    auto K::build(float alpha_, float beta_, Tensor const &a) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif
        return std::make_unique<K>(alpha_, beta_, a.dataType, a.elementsSize());
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing hardsigmoid operation on Nvidia GPU";
    }

#ifdef USE_CUDA
    constexpr static const char *TEMPLATE = R"~(
__device__ __forceinline__ static {0:} fn({0:} x) {{
    return {1:};
}}

extern "C" __global__ void kernel(
    {0:}       *__restrict__ y,
    {0:} const *__restrict__ x,
    size_t n
) {{
    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < n;
         tid += step)
        y[tid] = fn(x[tid]);
}}
    )~";
    auto K::lower(Resources &res) const -> RoutineWorkspace {
        using namespace runtime;

        std::string op = "";
        switch (dataType) {
            case DT::F32:
                op = fmt::format("fmaxf(0.f, fminf(1.f, fmaf({}, x, {})))", alpha, beta);
                break;
            case DT::F64:
                op = fmt::format("fmax(0.0, fmin(1.0, fma({}, x, {})))",
                                 static_cast<double>(alpha), static_cast<double>(beta));
                break;
            case DT::FP16:
                op = fmt::format("__hmax(CUDART_ZERO_FP16, __hmin(CUDART_ONE_FP16, (__float2half({}) * x + __float2half({}))))",
                                 alpha, beta);
                break;
            default:
                UNREACHABLE();
        }
        auto name = fmt::format("hardsigmoid_{}_{}_{}", dataType.name(), alpha, beta);
        auto code = fmt::format(TEMPLATE, nvrtc::dataType(dataType), op);
        return [h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "kernel"),
                params = cuda::ThreadsDistributer()(size)](
                   Resources &, void *, void const *const *inputs, void *const *outputs) {
            auto y = outputs[0];
            auto x = inputs[0];
            auto n = params.n;
            void *args[]{&y, &x, &n};
            h->launch(params.gridSize, 1, 1,
                      params.blockSize, 1, 1,
                      0, args);
        };
    }
#endif

}// namespace refactor::kernel

