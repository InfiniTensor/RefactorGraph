#include "cuda_kernel.hh"
#include <numeric>

namespace refactor::kernel {
    using K = RmsNormalizationCuda;
    using DT = DataType;

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
    auto K::lower(Resources &) const -> RoutineWorkspace {
        TODO("");
    }
#endif

}// namespace refactor::kernel
