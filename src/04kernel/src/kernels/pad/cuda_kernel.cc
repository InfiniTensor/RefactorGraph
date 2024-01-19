#include "cuda_kernel.hh"

#ifdef USE_CUDA
#include "../../generator/nvrtc_repo.h"
#include "kernel/cuda/threads_distributer.cuh"
#include <cuda_runtime.h>
#endif

namespace refactor::kernel {
    using K = PadCuda;

    K::PadCuda(PadInfo info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(PadInfo info) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif
        if (info.mode != PadType::Constant) {
            return nullptr;
        }
        return std::make_unique<K>(std::move(info));
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing Pad using CUDA";
    }

#ifdef USE_CUDA
    constexpr static const char *TEMPLATE = R"~(
#include "kernel/attributes/pad_info.h"

__device__ int WholeTensorOffset2PartTensorOffset(int tid,
                                                  PadInfo info) {{
    int offset = 0;
    for (int i = nDims - 1; i >= 0; --i) {{
        auto wholePos = tid % info.wholeNDim[i];
        auto pos = wholePos - info.begNum[i];
        // if pos belongs to pad range, then return -1
        if (pos < 0 || pos >= info.partNDim[i])
            return -1;
        tid = tid / info.wholeNDim[i];

        offset += pos * info.partStride[i];
    }}

    return offset;
}}
extern "C" __global__ void kernel(
    {0:}       *__restrict__ y,
    {0:} const *__restrict__ x,
    {0:} const *__restrict__ value,
    PadInfo info,
    size_t n
) {{
    auto const_value = info.have_value ? value[0] : static_cast<{0:}>(0);
    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < n;
         tid += step){{
            auto axis = WholeTensorOffset2PartTensorOffset(tid, info);
            y[tid] = axis < 0 ? const_value : x[tid];
        }}
}}
    )~";
    auto K::lower(Resources &res) const noexcept -> RoutineWorkspace {
        using namespace runtime;

        auto name = fmt::format("Pad_{}", info.type.name());
        auto code = fmt::format(TEMPLATE, nvrtc::dataType(info.type));
        return [info = this->info, h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "kernel"),
                params = cuda::ThreadsDistributer()(info.size)](
                   Resources &, void *, void const *const *inputs, void *const *outputs) {
            auto y = outputs[0];
            auto x = inputs[0];
            auto const_value = info.have_value ? inputs[2] : nullptr;
            auto n = params.n;
            void *args[]{&y, &x, &const_value, const_cast<PadInfo *>(&info), &n};
            h->launch(params.gridSize, 1, 1,
                      params.blockSize, 1, 1,
                      0, args);
        };
    }
#endif

}// namespace refactor::kernel
