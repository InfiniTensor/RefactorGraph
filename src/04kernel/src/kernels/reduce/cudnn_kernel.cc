#include "cudnn_kernel.hh"
#include "common.h"

namespace refactor::kernel {
    using K = ReduceCudnn;

    K::ReduceCudnn(
        decltype(dataType) dataType_,
        decltype(reduceType) reduceType_,
        decltype(axes) axes_,
        decltype(shape) shape_) noexcept
        : Kernel(),
          dataType(dataType_),
          reduceType(reduceType_),
          axes(std::move(axes_)),
          shape(std::move(shape_)) {}

    auto K::build(decltype(axes) axes_, ReduceType reduceType_, TensorRefs inputs_) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        auto const &x = inputs_[0].get();
        return x.dataType.isCpuNumberic()
                   ? std::make_unique<K>(x.dataType, reduceType_, std::move(axes_), x.shape)
                   : nullptr;
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing reduce operation using CUDNN";
    }

}// namespace refactor::kernel
