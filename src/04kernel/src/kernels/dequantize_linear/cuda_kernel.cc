#include "cuda_kernel.hh"

#ifdef USE_CUDA
#include "../../generator/nvrtc_repo.h"
#include "kernel/cuda/threads_distributer.cuh"
#endif

namespace refactor::kernel {
    using K = DequantizeLinearCuda;

    K::DequantizeLinearCuda(
        decltype(from) from_,
        decltype(to) to_,
        decltype(size) size_,
        decltype(withZeroPoint) withZeroPoint_) noexcept
        : Kernel(),
          from(from_),
          to(to_),
          size(size_),
          withZeroPoint(withZeroPoint_) {}

    auto K::build(TensorRefs const &inputs, Tensor const &output) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        auto const &x = inputs[0].get();
        if (inputs[1].get().elementsSize() != 1) {
            return nullptr;
        }
        return std::make_unique<K>(
            x.dataType,
            output.dataType,
            x.elementsSize(),
            inputs.size() > 2);
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing dequantize linear using Nvidia GPU";
    }

#ifdef USE_CUDA

    constexpr static const char *TEMPLATE = R"~(
extern "C" __global__ void kernel(
    {0:}       *__restrict__ y,
    {1:} const *__restrict__ x,
    {0:} const *__restrict__ scale_,
    {1:} const *__restrict__ zp_,
    size_t n
) {{
    auto zp = zp_ ? *zp_ : static_cast<{1:}>(0);
    auto scale = *scale_;
    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < n;
         tid += step) {{
        y[tid] = static_cast<{0:}>(x[tid] - zp) * scale;
    }}
}}
)~";

    auto K::lower(Resources &res) const -> RoutineWorkspace {
        using namespace runtime;

        auto name = fmt::format("DequantizeLinear{}->{}", from.name(), to.name());
        auto code = fmt::format(TEMPLATE, nvrtc::dataType(to), nvrtc::dataType(from));
        return [withZeroPoint = withZeroPoint,
                params = cuda::ThreadsDistributer()(size),
                h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "kernel")]//
            (Resources &, void *, void const *const *inputs, void *const *outputs) {
                auto y = outputs[0];
                auto x = inputs[0],
                     scale = inputs[1],
                     zp = withZeroPoint ? inputs[2] : nullptr;
                auto n = params.n;
                void *args[]{&y, &x, &scale, &zp, &n};
                h->launch(params.gridSize, 1, 1,
                          params.blockSize, 1, 1,
                          0, args);
            };
    }

#endif

}// namespace refactor::kernel
