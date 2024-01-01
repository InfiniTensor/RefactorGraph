#include "cublas_kernel.hh"

namespace refactor::kernel {
    using K = MatMulIntegerCublas;
    using DT = DataType;

    K::MatMulIntegerCublas(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(decltype(info) info) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return std::make_unique<K>(std::move(info));
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing MatMulInteger using CUBLAS";
    }

}// namespace refactor::kernel
