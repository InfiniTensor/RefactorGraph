#include "../../generator/cuda_code_repo.hh"
#include "cuda_kernel.hh"
#include "kernel/cuda/threads_distributer.cuh"
#include <sstream>

constexpr static const char *TEMPLATE = R"~(
struct Outputs {{
    char *const addr[{0:}];
}};

__global__ static void splitKernel(Outputs outputs, char const *data) {{
    using T = {1:};

    constexpr static unsigned int
        sum = {2:},
        segments[]{{{3:}}};
    auto data_ = reinterpret_cast<T const *>(data);

    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < {4:};
         tid += step) {{
        auto i = tid % sum * static_cast<unsigned int>(sizeof(T)), j = 0u;
        while (i >= segments[j]) i -= segments[j++];
        *reinterpret_cast<T *>(outputs.addr[j] + (tid / sum) * segments[j] + i) = data_[tid];
    }}
}}

extern "C" {{

void launchKernel(void const *data, void *const *outputs) {{
    splitKernel<<<{5:}, {6:}>>>(
        {{{7:}
        }},
        reinterpret_cast<char const *>(data));
}}

}}
)~";

namespace refactor::kernel {
    using namespace runtime;

    auto SplitCuda::lower(Resources &) const -> RoutineWorkspace {
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
