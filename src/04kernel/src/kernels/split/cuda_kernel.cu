#include "../../generator/cuda_code_repo.hh"
#include "cuda_kernel.hh"
#include "kernel/cuda/threads_distributer.cuh"
#include <sstream>

constexpr static const char *TEMPLATE = R"~(
struct Outputs {{
    char *const addr[{0:}];
}};

__global__ static void splitKernel(Outputs outputs, void const *input) {{
    using T = {1:};

    constexpr static unsigned int
        sum = {2:},
        segments[]{{{3:}}};
    auto src = reinterpret_cast<T const *>(input);

    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < {4:};
         tid += step) {{
        auto i = tid % sum * static_cast<unsigned int>(sizeof(T)), j = 0u;
        while (i >= segments[j]) i -= segments[j++];
        *reinterpret_cast<T *>(outputs.addr[j] + (tid / sum) * segments[j] + i) = src[tid];
    }}
}}

extern "C" {{

void launchKernel(void const *input, void *const *outputs) {{
    splitKernel<<<{5:}, {6:}>>>(
        {{{7:}
        }},
        input);
}}

}}
)~";

namespace refactor::kernel {
    using namespace runtime;

    auto SplitCuda::lower(Resources &) const noexcept -> RoutineWorkspace {
        if (info.blockCount == 1) {
            return [info = this->info](Resources &, void *, void const *const *inputs, void *const *outputs) {
                auto src = reinterpret_cast<uint8_t const *>(inputs[0]);
                for (auto i : range0_(info.segments.size())) {
                    auto size = info.segments[i];
                    cudaMemcpyAsync(outputs[i], src, size, cudaMemcpyDeviceToDevice);
                    src += size;
                }
            };
        }

        auto unit = info.unit(16);
        auto params = cuda::ThreadsDistributer()(info.blockCount * info.sum / unit);
        auto outputCount = info.segments.size();

        std::stringstream ss;
        for (auto seg : info.segments) {
            ss << seg << ", ";
        }
        auto segments = ss.str();

        ss.str("");
        for (auto i : range0_(outputCount)) {
            ss << std::endl
               << "            reinterpret_cast<char *>(outputs[" << i << "]), ";
        }
        auto castOutputs = ss.str();

        ss.str("");
        ss << "Split_" << info.blockCount << ',' << unit;
        for (auto seg : info.segments) {
            ss << ',' << seg;
        }
        auto name = ss.str();
        auto code = fmt::format(
            TEMPLATE,
            outputCount,                    // 0
            CudaCodeRepo::memCopyType(unit),// 1
            info.sum / unit,                // 2
            segments,                       // 3
            params.n,                       // 4
            params.gridSize,                // 5
            params.blockSize,               // 6
            castOutputs                     // 7
        );

        using Fn = void (*)(void const *, void *const *);
        auto function = reinterpret_cast<Fn>(CudaCodeRepo::compile_(name.c_str(), code.c_str(), "launchKernel"));
        return [function](Resources &, void *, void const *const *inputs, void *const *outputs) {
            function(inputs[0], outputs);
        };
    }

}// namespace refactor::kernel
