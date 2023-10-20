#include "cuda_kernel.hh"
#include "common.h"

namespace refactor::kernel {
    using K = TransposeCuda;

    K::TransposeCuda(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(Permutation const &perm, Tensor const &data) noexcept -> KernelBox {
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
        return "Performing transpose using CUDA";
    }

}// namespace refactor::kernel
