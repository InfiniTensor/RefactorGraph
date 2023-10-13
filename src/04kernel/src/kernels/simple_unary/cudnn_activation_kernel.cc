#include "cudnn_activation_kernel.hh"
#include "cudnn_activation_impl.hh"
#include <unordered_set>

namespace refactor::kernel {
    using K = ActivationCudnn;
    using DT = common::DataType;
    using Op = SimpleUnaryType;

    K::ActivationCudnn(Op type_, DT dataType_, size_t size_) noexcept
        : Kernel(), type(type_), dataType(dataType_), size(size_) {}

    auto K::build(Op op, Tensor const &a) noexcept -> KernelBox {
        static const std::unordered_set<decltype(DT::internal)> TYPE{
            DT::F32, DT::U8, DT::I8, DT::U16, DT::I16,
            DT::I32, DT::I64, DT::F64, DT::U32, DT::U64};
        static const std::unordered_set<Op> ARTHIMETIC{
            Op::Sigmoid, Op::Relu, Op::Tanh};

#ifndef USE_CUDA
        return nullptr;
#endif

        if (ARTHIMETIC.find(op) == ARTHIMETIC.end() ||
            TYPE.find(a.dataType) == TYPE.end()) {
            return nullptr;
        }
        return std::make_unique<K>(op, a.dataType, a.elementsSize());
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing activation using CUDNN";
    }
    auto K::lower() const noexcept -> Routine {
        return cudnn::lower(type, dataType, size);
    }

}// namespace refactor::kernel
