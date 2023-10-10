#include "cudnn_kernel.hh"
#include "common/error_handler.h"

namespace refactor::kernel {
    using K = ConvCudnn;

    K::ConvCudnn() noexcept : Kernel() {}

    auto K::build() noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return std::make_unique<K>();
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing conv using CUDNN";
    }
    auto K::lower() const noexcept -> Operation {
        TODO("");
    }

}// namespace refactor::kernel
