#include "cuda_kernel.hh"

namespace refactor::kernel {
    using K = ConcatCuda;

    K::ConcatCuda(SplitInfo info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(SplitInfo info) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif
        return std::make_unique<K>(std::move(info));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing split operation using CUDA";
    }

}// namespace refactor::kernel
