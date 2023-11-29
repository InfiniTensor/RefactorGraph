#include "../../generator/cuda_code_repo.hh"
#include "cuda_kernel.hh"
#include "kernel/cuda/threads_distributer.cuh"
#include <sstream>

constexpr static const char *TEMPLATE = R"~(
struct Inputs {{
    char const *const addr[{0:}];
}};

__global__ static void splitKernel(void *output, Inputs inputs) {{
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

extern "C" {{

void launchKernel(void const *const *inputs, void *output) {{
    splitKernel<<<{5:}, {6:}>>>(
        output,
        {{{7:}
        }});
}}

}}
)~";

namespace refactor::kernel {
    using namespace runtime;

    auto ConcatCuda::lower(Resources &) const noexcept -> RoutineWorkspace {
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
        auto name = ss.str();
        auto code = fmt::format(
            TEMPLATE,
            inputCount,                     // 0
            CudaCodeRepo::memCopyType(unit),// 1
            info.sum / unit,                // 2
            segments,                       // 3
            params.n,                       // 4
            params.gridSize,                // 5
            params.blockSize,               // 6
            castInputs                      // 7
        );

        using Fn = void (*)(void const *const *, void *);
        auto function = reinterpret_cast<Fn>(CudaCodeRepo::compile_(name.c_str(), code.c_str(), "launchKernel"));
        return [function](Resources &, void *, void const *const *inputs, void *const *outputs) {
            function(inputs, outputs[0]);
        };
    }

}// namespace refactor::kernel
