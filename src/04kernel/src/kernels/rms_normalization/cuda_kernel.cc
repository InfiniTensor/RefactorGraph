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

    auto K::build(float epsilon, Tensor const &x) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        if (!x.dataType.isFloat()) {
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
    // 1: blockDim
    // 2: T -> float
    // 3: T <- float
    constexpr static const char *TEMPLATE = R"~(
#include <cub/block/block_reduce.cuh>

extern "C" __global__ void kernel(
    {0:} *__restrict__ y,
    {0:} const *__restrict__ x,
    {0:} const *__restrict__ w,
    unsigned int blockSize,
    float epsilon) {{

    auto init = blockIdx.x * blockSize + threadIdx.x,
         step = blockDim.x;
    x += init;
    y += init;
    w += threadIdx.x;

    using BlockReduce = cub::BlockReduce<{0:}, {1:}>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ {0:} rms;

    auto acc = *x * *x;
    for (auto i = step; i < blockSize; i += step) {{
        acc += x[i] * x[i];
    }}
    acc = BlockReduce(tempStorage).Reduce(acc, cub::Sum());

    if (threadIdx.x == 0) {{
        rms = {3:}(rsqrt({2:}(acc) / blockSize + epsilon));
    }}
    __syncthreads();

    for (auto i = 0; i < blockSize; i += step) {{
        y[i] = x[i] * rms * w[i];
    }}
}}
)~";

    auto K::lower(Resources &) const -> RoutineWorkspace {
        using namespace runtime;

        std::stringstream ss;
        ss << "RmsNorm" << nvrtc::dataType(dataType) << blockSize;
        ss << ".cu";
        auto name = ss.str();
        auto blockDim = std::min(blockSize, 1024u);
        auto code = fmt::format(
            TEMPLATE,
            nvrtc::dataType(dataType),// 0
            blockDim,                 // 1
            // clang-format off
            dataType == DataType::F32  ? ""
          : dataType == DataType::F64  ? "static_cast<float>"
          : dataType == DataType::FP16 ? "__half2float"
          : dataType == DataType::BF16 ? "__bfloat162float"
          : UNREACHABLEX(const char*, "unreachable"),
            dataType == DataType::F32  ? ""
          : dataType == DataType::F64  ? ""
          : dataType == DataType::FP16 ? "__float2half"
          : dataType == DataType::BF16 ? "__float2bfloat16"
          : UNREACHABLEX(const char*, "unreachable")
            // clang-format on
        );

        return [h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "kernel"),
                epsilon_ = this->epsilon,
                blockCount = this->blockCount,
                blockSize_ = this->blockSize,
                blockDim]//
            (Resources &, void *, void const *const *inputs, void *const *outputs) {
                auto y = outputs[0];
                auto x = inputs[0];
                auto w = inputs[1];
                auto blockSize = blockSize_;
                auto epsilon = epsilon_;
                void *args[]{&y, &x, &w, &blockSize, &epsilon};
                h->launch(blockCount, 1, 1,
                          blockDim, 1, 1,
                          0, args);
            };
    }

#endif

}// namespace refactor::kernel
