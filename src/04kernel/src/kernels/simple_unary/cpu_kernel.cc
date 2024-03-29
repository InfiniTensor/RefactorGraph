﻿#include "cpu_kernel.hh"
#include <execution>
#include <unordered_set>

namespace refactor::kernel {
    using K = SimpleUnaryCpu;
    using Op = SimpleUnaryType;
    using DT = DataType;

    K::SimpleUnaryCpu(Op opType_, DT dataType_, size_t size_) noexcept
        : Kernel(), dataType(dataType_), opType(opType_), size(size_) {}

    auto K::build(Op op, Tensor const &a) noexcept -> KernelBox {
        static const std::unordered_set<Op> supportedOp{
            Op::Abs,
            Op::Relu,
            Op::Sqrt,
            Op::Sigmoid,
            Op::Tanh,
            Op::Neg,
            Op::Erf,
            Op::HardSwish,
            Op::Exp,
        };
        return supportedOp.contains(op) && a.dataType.isCpuNumberic()
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
        return "Performing unary operation on generic cpu";
    }

    template<class T> auto relu(T x) noexcept -> T { return x > 0 ? x : 0; }
    template<class T> auto sigmoid(T x) noexcept -> T {
        using M = std::conditional_t<sizeof(T) <= 4, float, double>;
        return static_cast<T>(1 / (1 + std::exp(-static_cast<M>(x))));
    }
    template<class T> auto convertSqrt(T x) noexcept -> T {
        using M = std::conditional_t<sizeof(T) <= 4, float, double>;
        return static_cast<T>(std::sqrt(static_cast<M>(x)));
    }
    template<class T> auto convertTanh(T x) noexcept -> T {
        using M = std::conditional_t<sizeof(T) <= 4, float, double>;
        return static_cast<T>(std::tanh(static_cast<M>(x)));
    }
    template<class T> auto hardswishFun(T x) noexcept -> T {
        auto mid = x / 6.f + .5f;
        return (mid <= 0)   ? 0
               : (1 <= mid) ? x
                            : x * mid;
    }
    auto copyForUnsigned(size_t n) noexcept -> Routine {
        return [n](runtime::Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            std::memcpy(outputs[0], inputs[0], n);
        };
    }

#define CASE(OP, T)                                                                                                       \
    case DT::T:                                                                                                           \
        return [n = this->size](runtime::Resources &, void *workspace, void const *const *inputs, void *const *outputs) { \
            using T_ = primitive<DT::T>::type;                                                                            \
            auto x = reinterpret_cast<T_ const *>(inputs[0]);                                                             \
            auto y = reinterpret_cast<T_ *>(outputs[0]);                                                                  \
            std::for_each_n(std::execution::par_unseq,                                                                    \
                            natural_t(0), n,                                                                              \
                            [y, x](auto i) { y[i] = OP(x[i]); });                                                         \
        }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        switch (opType) {
            case Op::Abs:
                switch (dataType) {
                    CASE(std::abs, F32);
                    CASE(std::abs, F64);
                    CASE(std::abs, I8);
                    CASE(std::abs, I16);
                    CASE(std::abs, I32);
                    CASE(std::abs, I64);
                    case DT::U8:
                    case DT::U16:
                    case DT::U32:
                    case DT::U64:
                        return copyForUnsigned(size * dataType.size());
                    default:
                        UNREACHABLE();
                }
            case Op::Relu:
                switch (dataType) {
                    CASE(relu, F32);
                    CASE(relu, F64);
                    CASE(relu, I8);
                    CASE(relu, I16);
                    CASE(relu, I32);
                    CASE(relu, I64);
                    case DT::U8:
                    case DT::U16:
                    case DT::U32:
                    case DT::U64:
                        return copyForUnsigned(size * dataType.size());
                    default:
                        UNREACHABLE();
                }
            case Op::Sqrt:
                switch (dataType) {
                    CASE(std::sqrt, F32);
                    CASE(std::sqrt, F64);
                    CASE(convertSqrt, I8);
                    CASE(convertSqrt, I16);
                    CASE(convertSqrt, I32);
                    CASE(convertSqrt, I64);
                    CASE(convertSqrt, U8);
                    CASE(convertSqrt, U16);
                    CASE(convertSqrt, U32);
                    CASE(convertSqrt, U64);
                    default:
                        UNREACHABLE();
                }
            case Op::Sigmoid:
                switch (dataType) {
                    CASE(sigmoid, F32);
                    CASE(sigmoid, F64);
                    CASE(sigmoid, I8);
                    CASE(sigmoid, I16);
                    CASE(sigmoid, I32);
                    CASE(sigmoid, I64);
                    CASE(sigmoid, U8);
                    CASE(sigmoid, U16);
                    CASE(sigmoid, U32);
                    CASE(sigmoid, U64);
                    default:
                        UNREACHABLE();
                }
            case Op::Tanh:
                switch (dataType) {
                    CASE(std::tanh, F32);
                    CASE(std::tanh, F64);
                    CASE(convertTanh, I8);
                    CASE(convertTanh, I16);
                    CASE(convertTanh, I32);
                    CASE(convertTanh, I64);
                    CASE(convertTanh, U8);
                    CASE(convertTanh, U16);
                    CASE(convertTanh, U32);
                    CASE(convertTanh, U64);
                    default:
                        UNREACHABLE();
                }
            case Op::Neg:
                switch (dataType) {
                    CASE(-, F32);
                    CASE(-, F64);
                    CASE(-, I8);
                    CASE(-, I16);
                    CASE(-, I32);
                    CASE(-, I64);
                    default:
                        UNREACHABLE();
                }
            case Op::Erf:
                switch (dataType) {
                    CASE(std::erf, F32);
                    CASE(std::erf, F64);
                    CASE(std::erf, I8);
                    CASE(std::erf, I16);
                    CASE(std::erf, I32);
                    CASE(std::erf, I64);
                    CASE(std::erf, U8);
                    CASE(std::erf, U16);
                    CASE(std::erf, U32);
                    CASE(std::erf, U64);
                    default:
                        UNREACHABLE();
                }
            case Op::HardSwish:
                switch (dataType) {
                    CASE(hardswishFun, F32);
                    CASE(hardswishFun, F64);
                    default:
                        UNREACHABLE();
                }
            case Op::Exp:
                switch (dataType) {
                    CASE(std::exp, F32);
                    CASE(std::exp, F64);
                    default:
                        UNREACHABLE();
                }
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
