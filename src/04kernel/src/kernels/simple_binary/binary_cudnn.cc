#include "binary_cudnn.hh"
#include <execution>
#include <unordered_set>

namespace refactor::kernel {
    using K = BinaryCudnn;
    using Op = SimpleBinaryType;
    using DT = DataType;

    K::BinaryCudnn(Op opType_, DT dataType_, std::vector<int> aDims_, std::vector<int> bDims_, std::vector<int> cDims_) noexcept
        : Kernel(), dataType(dataType_), opType(opType_), aDims(aDims_), bDims(bDims_), cDims(cDims_) {}

    auto K::build(Op op, Tensor const &a, Tensor const &b, Tensor const &c) noexcept -> KernelBox {
        static const std::unordered_set<decltype(DT::internal)> TYPE{
            DT::F32, DT::F64, DT::FP16, DT::I8, DT::I32, DT::U8, DT::BF16,
            DT::I64};
        static const std::unordered_set<Op> ARTHIMETIC{
            Op::Add, Op::Sub, Op::Mul};

#ifndef USE_CUDA
        return nullptr;
#endif

        if (a.dataType != b.dataType ||
            ARTHIMETIC.find(op) == ARTHIMETIC.end() ||
            TYPE.find(a.dataType) == TYPE.end() ||
            // At least one of a,b should have the same shape as c
            (a.shape != c.shape && b.shape != c.shape) ||
            // Sub only supports brocasting b
            (a.shape != c.shape && op == Op::Sub) ||
            // Cudnn binary op only supports up to 5D
            !((a.rank() == 5 && b.rank() == 5) || (a.rank() <= 4 && b.rank() <= 4))) {
            return nullptr;
        }

        auto shape2IntVec = [](Shape shape) -> std::vector<int> {
            std::vector<int> intVector;
            intVector.reserve(shape.size());
            for (const uint32_t &element : shape) {
                intVector.push_back(static_cast<int>(element));
            }
            return intVector;
        };

        return std::make_unique<K>(op, a.dataType, shape2IntVec(a.shape), shape2IntVec(b.shape), shape2IntVec(c.shape));
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing element-wise op of 2 tensors with CUDNN";
    }


}// namespace refactor::kernel
