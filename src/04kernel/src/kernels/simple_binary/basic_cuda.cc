#include "basic_cuda.hh"

#ifdef USE_CUDA
#include "../../generator/nvrtc_repo.h"
#include "codegen.cuh"
#include "kernel/cuda/threads_distributer.cuh"
#include <cuda_runtime.h>
#endif

namespace refactor::kernel {
    using K = BinaryBasicCuda;
    using Op = SimpleBinaryType;
    using DT = DataType;

    K::BinaryBasicCuda(Op opType_, DT dataType_, Broadcaster b) noexcept
        : Kernel(),
          dataType(dataType_),
          opType(opType_),
          broadcaster(std::move(b)) {}

    auto K::build(Op op, Tensor const &a, Tensor const &b) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return a.dataType == b.dataType
                   ? std::make_unique<K>(op, a.dataType, Broadcaster({a, b}))
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
        return "Performing binary operation of 2 tensors on Nvidai GPU";
    }

#ifdef USE_CUDA

    constexpr static const char *TEMPLATE = R"~(
struct Strides {{
    unsigned int s[3 * {2:}];
}};

__device__ __forceinline__ static {0:} fn({0:} a, {0:} b) {{
    return {1:};
}}

extern "C" __global__ void kernel({0:} *c, {0:} const *a, {0:} const *b, Strides strides, size_t n) {{
    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < n;
         tid += step) {{
        unsigned int ia = 0, ib = 0;
        auto rem = tid;
        for (auto i = 0; i < {2:}; ++i) {{
            auto dim = strides.s + 3 * i;
            auto quot = rem / dim[2];
            ia += dim[0] * quot;
            ib += dim[1] * quot;
            rem %= dim[2];
        }}
        c[tid] = fn(a[ia], b[ib]);
    }}
}}
)~";

    auto K::lower(Resources &res) const -> RoutineWorkspace {
        using namespace runtime;
        auto rank = broadcaster.strides.size() / 3;
        auto name = fmt::format("binary{}_{}_{}", rank, dataType.name(), opName(opType));
        auto code = fmt::format(TEMPLATE,
                                nvrtc::dataType(dataType),
                                op(opType, dataType),
                                rank);
        return [h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "kernel"),
                strides = broadcaster.strides,
                params = cuda::ThreadsDistributer()(broadcaster.outputsCount)](
                   Resources &, void *, void const *const *inputs, void *const *outputs) {
            auto c = outputs[0];
            auto a = inputs[0],
                 b = inputs[1];
            auto n = params.n;
            void *args[]{&c, &a, &b, const_cast<dim_t *>(strides.data()), &n};
            CUDA_ASSERT(cuLaunchKernel(
                h->kernel(),
                params.gridSize, 1, 1,
                params.blockSize, 1, 1,
                0, nullptr, args, nullptr));
        };
    }

#endif

}// namespace refactor::kernel
