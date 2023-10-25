#include "cuda_kernel.hh"
#include "common.h"

namespace refactor::kernel {
    using K = TransposeCuda;
    using Info = TransposeInfo;

    K::TransposeCuda(DataType dataType_, Info info_) noexcept
        : Kernel(), dataType(dataType_), info(std::move(info_)) {}

    auto K::build(DataType dataType, Info info) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif
        return std::make_unique<K>(dataType, std::move(info));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing transpose operation using CUDA";
    }

}// namespace refactor::kernel
