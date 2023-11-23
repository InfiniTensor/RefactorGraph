#include "../../generator/cuda_code_repo.hh"
#include "cuda_kernel.hh"
#include "kernel/cuda/threads_distributer.cuh"
#include <sstream>

constexpr static const char *TEMPLATE = R"~(
__global__ static void splitKernel({0:}char const *data) {{

    constexpr static unsigned int
        n = {1:},
        sum = {2:},
        sub = {3:},
        segments[]{{{4:}}};
    char *outputs[]{{{5:}}};

    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < n;
         tid += step) {{
        auto i = tid % sum, j = i * sub, k = 0u;
        while (j >= segments[k]) j -= segments[k++];
        memcpy(outputs[k] + (tid / sum) * segments[k] + j, data + tid * sub, sub);
    }}
}}

extern "C" {{

void launchKernel(void const *data, void *const *outputs) {{
    splitKernel<<<{6:}, {7:}>>>({8:}
        reinterpret_cast<char const*>(data));
}}

}}
)~";

namespace refactor::kernel {
    using namespace runtime;

    auto SplitCuda::lower(Resources &) const noexcept -> RoutineWorkspace {
        auto sub = std::min(info.submultiple(), 16u);
        auto params = cuda::ThreadsDistributer()(info.blockCount * info.sum / sub);
        auto outputCount = info.segments.size();

        std::stringstream ss;
        for (auto i : range0_(outputCount)) {
            ss << "char *output" << i << ", ";
        }
        auto s0 = ss.str();

        ss.str("");
        for (auto seg : info.segments) {
            ss << seg << ", ";
        }
        auto s5 = ss.str();

        ss.str("");
        for (auto i : range0_(outputCount)) {
            ss << "output" << i << ", ";
        }
        auto s6 = ss.str();

        ss.str("");
        for (auto i : range0_(outputCount)) {
            ss << std::endl
               << "        reinterpret_cast<char *>(outputs[" << i << "]), ";
        }
        auto s9 = ss.str();

        auto code = fmt::format(
            TEMPLATE,
            s0,              // 0
            params.n,        // 1
            info.sum / sub,  // 2
            sub,             // 3
            s5,              // 4
            s6,              // 5
            params.gridSize, // 6
            params.blockSize,// 7
            s9               // 8
        );

        using Fn = void (*)(void const *, void *const *);
        auto function = reinterpret_cast<Fn>(CudaCodeRepo().compile("split", code.c_str(), "launchKernel"));
        return [function](Resources &, void *, void const *const *inputs, void *const *outputs) {
            function(inputs[0], outputs);
        };
    }

}// namespace refactor::kernel
