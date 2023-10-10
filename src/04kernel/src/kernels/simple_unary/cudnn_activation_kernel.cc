#include "cudnn_activation_kernel.hh"
#include "common/error_handler.h"

namespace refactor::kernel {
    using K = ActivationCudnn;
    using Ty = SimpleUnaryType;

    K::ActivationCudnn(SimpleUnaryType type_) noexcept
        : Kernel(), type(type_) {}

    auto K::build(SimpleUnaryType type, Tensor const &) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return std::make_unique<K>(type);
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing activation using CUDNN";
    }
    auto K::lower() const noexcept -> Operation {
        TODO("");
    }

}// namespace refactor::kernel
