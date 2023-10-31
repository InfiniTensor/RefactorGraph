#include "basic_cuda.hh"
#include <execution>

namespace refactor::kernel {
    using K = BinaryBasicCuda;
    using Op = SimpleBinaryType;
    using DT = DataType;

    K::BinaryBasicCuda(Op opType_, DT dataType_, Broadcaster b) noexcept
        : Kernel(),
          dataType(dataType_),
          opType(opType_),
          broadcaster(std::move(b)) {}

    auto K::build(Op op, Tensor const &a, Tensor const &b) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return a.dataType.isCpuNumberic() && a.dataType == b.dataType
                   ? std::make_unique<K>(op, a.dataType, Broadcaster({a, b}))
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
        return "Performing binary operation of 2 tensors on Nvidai GPU";
    }

}// namespace refactor::kernel
