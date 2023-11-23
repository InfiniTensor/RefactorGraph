#include "../../generator/cuda_code_repo.hh"
#include "cuda_kernel.hh"
#include "kernel/cuda/threads_distributer.cuh"
#include <sstream>

constexpr static const char *TEMPLATE = R"~(
__global__ static void splitKernel({0:}char const *data) {{
    using T = {1:};

    constexpr static unsigned int
        n = {2:},
        sum = {3:},
        segments[]{{{4:}}};
    auto data_ = reinterpret_cast<T const *>(data);
    char *outputs[]{{{5:}}};

    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < n;
         tid += step) {{
        auto i = tid % sum, j = i * static_cast<unsigned int>(sizeof(T)), k = 0u;
        while (j >= segments[k]) j -= segments[k++];
        *reinterpret_cast<T *>(outputs[k] + (tid / sum) * segments[k] + j) = data_[tid];
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

        ss.str("");
        ss << "Split" << outputCount;
        for (auto seg : info.segments) {
            ss << '_' << seg;
        }
        auto name = ss.str();
        auto code = fmt::format(
            TEMPLATE,
            s0,                            // 0
            CudaCodeRepo::memCopyType(sub),// 1
            params.n,                      // 2
            info.sum / sub,                // 3
            s5,                            // 4
            s6,                            // 5
            params.gridSize,               // 6
            params.blockSize,              // 7
            s9                             // 8
        );

        using Fn = void (*)(void const *, void *const *);
        auto function = reinterpret_cast<Fn>(CudaCodeRepo::compile_(name.c_str(), code.c_str(), "launchKernel"));
        return [function](Resources &, void *, void const *const *inputs, void *const *outputs) {
            function(inputs[0], outputs);
        };
    }

}// namespace refactor::kernel
