#include "binary_cuda.hh"
#include <execution>
#include <unordered_set>

namespace refactor::kernel {
    using K = BinaryCuda;
    using Op = SimpleBinaryType;
    using DT = DataType;

    K::BinaryCuda(Op opType_, DT dataType_, size_t size_, bool constB_) noexcept
        : Kernel(), dataType(dataType_), opType(opType_), size(size_), constB(constB_) {}

    auto K::build(Op op, Tensor const &a, Tensor const &b) noexcept -> KernelBox {
        static const std::unordered_set<decltype(DT::internal)> TYPE{
            DT::F32, DT::U8, DT::I8, DT::U16, DT::I16,
            DT::I32, DT::I64, DT::F64, DT::U32, DT::U64};
        static const std::unordered_set<Op> ARTHIMETIC{
            Op::Add, Op::Sub, Op::Mul, Op::Div};

#ifndef USE_CUDA
        return nullptr;
#endif

        if (a.dataType != b.dataType ||
            ARTHIMETIC.find(op) == ARTHIMETIC.end() ||
            TYPE.find(a.dataType) == TYPE.end()) {
            return nullptr;
        }

        bool constantB = b.rank() == 0 || b.shape == Shape{1};

        // Support 1 to 1 and constant b
        // TODO: add other broadcast if needed in the future
        if (!(a.shape == b.shape || constantB)) {
            return nullptr;
        }

        return std::make_unique<K>(op, a.dataType, a.elementsSize(), constantB);
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing add/sub/mul/div of 2 tensors of same shape or constant on nvidia gpu";
    }

}// namespace refactor::kernel
