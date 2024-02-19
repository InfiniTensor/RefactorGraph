#include "cuda_kernel.hh"

namespace refactor::kernel {
    using K = AttentionCuda;

    K::AttentionCuda(decltype(info) info_) noexcept
        : Kernel(), info(info_) {}

    auto K::build(decltype(info) info) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return std::make_unique<K>(info);
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing multihead attention on Nvidia gpu";
    }

}// namespace refactor::kernel
