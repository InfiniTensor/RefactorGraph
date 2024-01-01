#include "cublas_kernel.hh"

namespace refactor::kernel {
    using K = MatMulCublas;
    using DT = DataType;

    K::MatMulCublas(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(decltype(info) info) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return info.dataType.isIeee754() || info.dataType == DT::I8
                   ? std::make_unique<K>(std::move(info))
                   : nullptr;
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing MatMul using CUBLAS";
    }

}// namespace refactor::kernel
