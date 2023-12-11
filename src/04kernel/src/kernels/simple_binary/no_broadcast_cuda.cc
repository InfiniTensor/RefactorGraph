#include "no_broadcast_cuda.hh"

#ifdef USE_CUDA
#include "../../generator/nvrtc_repo.h"
#include "codegen.cuh"
#include "kernel/cuda/threads_distributer.cuh"
#include <cuda_runtime.h>
#endif

namespace refactor::kernel {
    using K = Binary11Cuda;
    using Op = SimpleBinaryType;
    using DT = DataType;

    K::Binary11Cuda(Op opType_, DT dataType_, size_t size_) noexcept
        : Kernel(), dataType(dataType_), opType(opType_), size(size_) {}

    auto K::build(Op op, Tensor const &a, Tensor const &b) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return (a.dataType.isNumberic() || a.dataType.isBool()) &&
                       a.dataType == b.dataType &&
                       a.shape == b.shape
                   ? std::make_unique<K>(op, a.dataType, a.elementsSize())
                   : nullptr;
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing binary operation of 2 tensors with same shape on Nvidia GPU";
    }

#ifdef USE_CUDA

    constexpr static const char *TEMPLATE = R"~(
__device__ __forceinline__ static {0:} fn({0:} a, {0:} b) {{
    return {1:};
}}

extern "C" __global__ void kernel({0:} *c, {0:} const *a, {0:} const *b, size_t n) {{
    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < n;
         tid += step)
        c[tid] = fn(a[tid], b[tid]);
}}
)~";

    auto K::lower(Resources &res) const -> RoutineWorkspace {
        using namespace runtime;
        auto name = fmt::format("binary_{}_{}", dataType.name(), opName(opType));
        auto code = fmt::format(TEMPLATE, nvrtc::dataType(dataType), op(opType, dataType));
        return [h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "kernel"),
                params = cuda::ThreadsDistributer()(size)](
                   Resources &, void *, void const *const *inputs, void *const *outputs) {
            auto c = outputs[0];
            auto a = inputs[0],
                 b = inputs[1];
            auto n = params.n;
            void *args[]{&c, &a, &b, &n};
            CUDA_ASSERT(cuLaunchKernel(
                h->kernel(),
                params.gridSize, 1, 1,
                params.blockSize, 1, 1,
                0, nullptr, args, nullptr));
        };
    }

#endif

}// namespace refactor::kernel
