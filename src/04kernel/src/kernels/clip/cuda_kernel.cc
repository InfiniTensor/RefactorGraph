#include "cuda_kernel.hh"

namespace refactor::kernel {
    using K = ClipCuda;

    K::ClipCuda(decltype(dataType) dt,
                decltype(size) size_,
                decltype(hasMax) hasMax_) noexcept
        : dataType(dt), size(size_), hasMax(hasMax_) {
    }

    auto K::build(Tensor const &data, bool hasMax) noexcept -> KernelBox {
        return data.dataType.isCpuNumberic()
                   ? std::make_unique<K>(data.dataType, data.elementsSize(), hasMax)
                   : nullptr;
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing clip operation on Nvidia GPU";
    }

}// namespace refactor::kernel
