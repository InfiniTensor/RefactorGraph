#include "cuda_kernel.hh"

#ifdef USE_CUDA
#include "../../generator/nvrtc_repo.h"
#include "kernel/cuda/threads_distributer.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>
#endif

namespace refactor::kernel {
    using K = ConcatCuda;

    K::ConcatCuda(SplitInfo info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(SplitInfo info) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif
        return std::make_unique<K>(std::move(info));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing split operation using CUDA";
    }

#ifdef USE_CUDA

    constexpr static const char *TEMPLATE = R"~(
struct Inputs {{
    char const *const addr[{0:}];
}};

extern "C" __global__ void concatKernel(void *output, Inputs inputs) {{
    using T = {1:};

    constexpr static unsigned int
        sum = {2:},
        segments[]{{{3:}}};
    auto dst = reinterpret_cast<T *>(output);

    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < {4:};
         tid += step) {{
        auto i = tid % sum * static_cast<unsigned int>(sizeof(T)), j = 0u;
        while (i >= segments[j]) i -= segments[j++];
        dst[tid] = *reinterpret_cast<T const *>(inputs.addr[j] + (tid / sum) * segments[j] + i);
    }}
}}
)~";

    auto K::lower(Resources &) const -> RoutineWorkspace {
        using namespace runtime;
        if (info.blockCount == 1) {
            return [info = this->info](Resources &, void *, void const *const *inputs, void *const *outputs) {
                auto dst = reinterpret_cast<uint8_t *>(outputs[0]);
                for (auto i : range0_(info.segments.size())) {
                    auto size = info.segments[i];
                    cudaMemcpyAsync(dst, inputs[i], size, cudaMemcpyDeviceToDevice);
                    dst += size;
                }
            };
        }

        auto unit = info.unit(16);
        auto params = cuda::ThreadsDistributer()(info.blockCount * info.sum / unit);
        auto inputCount = info.segments.size();

        std::stringstream ss;
        for (auto seg : info.segments) {
            ss << seg << ", ";
        }
        auto segments = ss.str();

        ss.str("");
        for (auto i : range0_(inputCount)) {
            ss << std::endl
               << "            reinterpret_cast<char const *>(inputs[" << i << "]), ";
        }
        auto castInputs = ss.str();

        ss.str("");
        ss << "Concat_" << info.blockCount << ',' << unit;
        for (auto seg : info.segments) {
            ss << ',' << seg;
        }
        ss << ".cu";
        auto name = ss.str();
        auto code = fmt::format(
            TEMPLATE,
            inputCount,              // 0
            nvrtc::memCopyType(unit),// 1
            info.sum / unit,         // 2
            segments,                // 3
            params.n                 // 4
        );

        return [h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "concatKernel"),
                params](Resources &, void *, void const *const *inputs, void *const *outputs) {
            void *args[]{const_cast<void **>(outputs), const_cast<void **>(inputs)};
            CUDA_ASSERT(cuLaunchKernel(
                h->kernel(),
                params.gridSize, 1, 1,
                params.blockSize, 1, 1,
                0, nullptr, args, nullptr));
        };
    }

#endif

}// namespace refactor::kernel
