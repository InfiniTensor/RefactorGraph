#include "cpu_kernel.hh"
#include "common.h"
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
        return "Performing unary operation on generic cpu";
    }

#define CASE(OP, T)                                                                          \
    case DT::T:                                                                              \
        return [n = this->size](runtime::Resources &, void const **inputs, void **outputs) { \
            using T_ = primitive_t<DT::T>::type;                                             \
            auto x = static_cast<T_ const *>(inputs[0]);                                     \
            auto y = static_cast<T_ *>(outputs[0]);                                          \
            std::for_each_n(std::execution::par_unseq,                                       \
                            natural_t(0), n,                                                 \
                            [y, x](auto i) { y[i] = OP(x[i]); });                            \
        }

    auto copyForUnsigned(size_t n) noexcept -> Routine {
        return [n](runtime::Resources &, void const **inputs, void **outputs) {
            std::memcpy(outputs[0], inputs[0], n);
        };
    }
    template<class T> auto relu(T x) noexcept -> T { return x > 0 ? x : 0; }
    template<class M, class T> auto sigmoid(T x) noexcept -> T {
        return static_cast<T>(1 / (1 + std::exp(-static_cast<M>(x))));
    }
    template<class M, class T> auto convertTanh(T x) noexcept -> T {
        return static_cast<T>(std::tanh(static_cast<M>(x)));
    }

    auto K::lower() const noexcept -> Routine {
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
            case Op::Sigmoid:
                switch (dataType) {
                    CASE(sigmoid<float>, F32);
                    CASE(sigmoid<double>, F64);
                    CASE(sigmoid<float>, I8);
                    CASE(sigmoid<float>, I16);
                    CASE(sigmoid<double>, I32);
                    CASE(sigmoid<double>, I64);
                    CASE(sigmoid<float>, U8);
                    CASE(sigmoid<float>, U16);
                    CASE(sigmoid<double>, U32);
                    CASE(sigmoid<double>, U64);
                    default:
                        UNREACHABLE();
                }
            case Op::Tanh:
                switch (dataType) {
                    CASE(std::tanh, F32);
                    CASE(std::tanh, F64);
                    CASE(convertTanh<float>, I8);
                    CASE(convertTanh<float>, I16);
                    CASE(convertTanh<double>, I32);
                    CASE(convertTanh<double>, I64);
                    CASE(convertTanh<float>, U8);
                    CASE(convertTanh<float>, U16);
                    CASE(convertTanh<double>, U32);
                    CASE(convertTanh<double>, U64);
                    default:
                        UNREACHABLE();
                }
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
