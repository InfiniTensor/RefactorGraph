#include "cudnn_activation_kernel.hh"
#include "kernel/collectors/simple_unary.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"
#include <unordered_set>

namespace refactor::kernel {
    using K = ActivationCudnn;
    using DT = DataType;
    using Op = SimpleUnaryType;

    K::ActivationCudnn(Op type_, DT dataType_, int size_) noexcept
        : Kernel(), type(type_), dataType(dataType_), size(size_) {}

    auto K::build(Op op, Tensor const &a) noexcept -> KernelBox {
        static const std::unordered_set<Op> ARTHIMETIC{Op::Sigmoid, Op::Relu, Op::Tanh};

#ifndef USE_CUDA
        return nullptr;
#endif

        return ARTHIMETIC.contains(op) && a.dataType.isCpuNumberic()
                   ? std::make_unique<K>(op, a.dataType, static_cast<int>(a.elementsSize()))
                   : nullptr;
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing activation using CUDNN";
    }

}// namespace refactor::kernel
