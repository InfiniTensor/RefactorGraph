#include "no_broadcast_cuda.hh"
#include <execution>

namespace refactor::kernel {
    using K = Binary11Cuda;
    using Op = SimpleBinaryType;
    using DT = DataType;

    K::Binary11Cuda(Op opType_, DT dataType_, size_t size_) noexcept
        : Kernel(), dataType(dataType_), opType(opType_), size(size_) {}

    auto K::build(Op op, Tensor const &a, Tensor const &b) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return a.dataType.isCpuNumberic() && a.dataType == b.dataType && a.shape == b.shape
                   ? std::make_unique<K>(op, a.dataType, a.elementsSize())
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
        return "Performing binary operation of 2 tensors with same shape on Nvidia GPU";
    }

}// namespace refactor::kernel
