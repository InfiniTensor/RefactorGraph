#include "cuda_kernel.hh"

namespace refactor::kernel {
    using K = DynamicQuantizeLinearCuda;

    K::DynamicQuantizeLinearCuda(decltype(size) size_) noexcept
        : Kernel(), size(size_) {}

    auto K::build(decltype(size) size) noexcept -> KernelBox {
        return std::make_unique<K>(size);
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing dynamic quantize linear using Nvidia GPU";
    }

}// namespace refactor::kernel
