#include "cuda_kernel.hh"
#include <execution>
#include <unordered_set>

namespace refactor::kernel {
    using K = SimpleUnaryCuda;
    using Op = SimpleUnaryType;
    using DT = DataType;

    K::SimpleUnaryCuda(Op opType_, DT dataType_, size_t size_) noexcept
        : Kernel(), dataType(dataType_), opType(opType_), size(size_) {}

    auto K::build(Op op, Tensor const &a) noexcept -> KernelBox {
        static const std::unordered_set<Op> supportedOp{
            Op::Abs,
            Op::Relu,
            Op::Sqrt,
            Op::Sigmoid,
            Op::Tanh,
        };
        if (supportedOp.find(op) == supportedOp.end()) {
            return nullptr;
        }
        if (!a.dataType.isCpuNumberic()) {
            return nullptr;
        }
        return std::make_unique<K>(op, a.dataType, a.elementsSize());
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing unary operation on Nvidia GPU";
    }

}// namespace refactor::kernel
