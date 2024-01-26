#include "cuda_kernel.hh"

namespace refactor::kernel {
    using K = PadCuda;

    K::PadCuda(PadInfo info_, PadType mode_, size_t value_) noexcept
        : Kernel(), info(std::move(info_)), mode(mode_), valueLength(value_) {}

    auto K::build(PadInfo info, PadType mode, std::optional<std::reference_wrapper<Tensor const>> value_) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif
        if (mode != PadType::Constant) {
            return nullptr;
        }
        size_t value = value_ ? value_->get().dataType.size() : 0;
        info.reform(16);
        return std::make_unique<K>(std::move(info), mode, value);
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing Pad using CUDA";
    }

}// namespace refactor::kernel
