#include "nccl_kernel.hh"

namespace refactor::kernel {
    using K = AllReduceNccl;
    using DT = DataType;

    K::AllReduceNccl(AllReduceType opType_, DT dataType_, size_t size_) noexcept
        : opType(opType_), dataType(dataType_), size(size_) {}

    auto K::build(AllReduceType opType_, Tensor const &input, Tensor const &output) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif
        if (input.elementsSize() != output.elementsSize() ||
            input.dataType != output.dataType) {
            return nullptr;
        }

        return std::make_unique<K>(opType_, input.dataType, input.elementsSize());
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing AllReduce using NCCL";
    }

}// namespace refactor::kernel