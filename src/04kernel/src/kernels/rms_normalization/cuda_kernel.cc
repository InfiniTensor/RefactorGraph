#include "cuda_kernel.hh"
#include <numeric>

#ifdef USE_CUDA
#include "../../generator/nvrtc_repo.h"
#include <cuda_runtime.h>
#include <sstream>
#endif

namespace refactor::kernel {
    using K = RmsNormalizationCuda;

    K::RmsNormalizationCuda(
        decltype(epsilon) epsilon_,
        decltype(dataType) dataType_,
        decltype(blockCount) blockCount_,
        decltype(blockSize) blockSize_) noexcept
        : Kernel(),
          epsilon(epsilon_),
          dataType(dataType_),
          blockCount(blockCount_),
          blockSize(blockSize_) {}

    auto K::build(float epsilon, TensorRefs inputs) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        auto const &x = inputs[0].get();
        auto const &w = inputs[1].get();
        if (!x.dataType.isFloat() || x.dataType != w.dataType) {
            return nullptr;
        }
        auto it = x.shape.rbegin();
        dim_t blockSize = *it++;
        dim_t blockCount = std::accumulate(it, x.shape.rend(), 1, std::multiplies());
        return std::make_unique<K>(epsilon, x.dataType, blockCount, blockSize);
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing rms normalization using CUDA";
    }

#ifdef USE_CUDA

    // 0: data type
    // 1: block size
    // 2: epsilon cast
    constexpr static const char *TEMPLATE = R"~(
#include <cub/cub.cuh>

static __device__ __forceinline__ {0:} squareSum({0:} a, {0:} b) {{
    return a * a + b * b;
}}

extern "C" __global__ void kernel(
    {0:} *__restrict__ const y,
    {0:} const *__restrict__ const x,
    {0:} const *__restrict__ const w,
    float epsilon_) {{

    auto epsilon = {2:}(epsilon_);
    x += blockIdx.x * blockDim.x + threadIdx.x;
    y += blockIdx.x * blockDim.x + threadIdx.x;;
    w += threadIdx.x;

    using BlockReduce = cub::BlockReduce<{0:}, {1:}>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ {0:} rms;
    auto acc = BlockReduce(tempStorage).Reduce(*x, squareSum);
    if (threadIdx.x == 0) {{
        rms = rsqrt(acc / blockDim.x + epsilon);
    }}
    __syncthreads();

    *y = *x * rms * *w;
}}
)~";

    auto K::lower(Resources &) const -> RoutineWorkspace {
        using namespace runtime;

        std::stringstream ss;
        ss << "RmsNorm" << nvrtc::dataType(dataType) << blockSize;
        ss << ".cu";
        auto name = ss.str();
        auto code = fmt::format(
            TEMPLATE,
            nvrtc::dataType(dataType),// 0
            blockSize,                // 1
            // clang-format off
            dataType == DataType::F32  ? ""
          : dataType == DataType::F64  ? "static_cast<float>"
          : dataType == DataType::FP16 ? "__half2float"
          : dataType == DataType::BF16 ? "__bfloat162float"
          : UNREACHABLEX(const char*, "unreachable")
            // clang-format on
        );

        return [h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "kernel"),
                epsilon_ = this->epsilon,
                blockCount = this->blockCount,
                blockSize = this->blockSize]//
            (Resources &, void *, void const *const *inputs, void *const *outputs) {
                auto y = outputs[0];
                auto x = inputs[0];
                auto w = inputs[1];
                auto epsilon = epsilon_;
                void *args[]{&y, &x, &w, &epsilon};
                h->launch(blockCount, 1, 1,
                          blockSize, 1, 1,
                          0, args);
            };
    }

#endif

}// namespace refactor::kernel
