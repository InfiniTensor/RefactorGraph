#include "cuda_kernel.hh"
#include "refactor/common.h"

namespace refactor::kernel {
    using K = GatherCuda;

    K::GatherCuda(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(Tensor const &, Tensor const &) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return nullptr;
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing gather using CUDA";
    }

}// namespace refactor::kernel
