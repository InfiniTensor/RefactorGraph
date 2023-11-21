#include "no_broadcast_cpu.hh"
#include <execution>
#include <unordered_set>

namespace refactor::kernel {
    using K = Binary11Cpu;
    using Op = SimpleBinaryType;
    using DT = DataType;

    K::Binary11Cpu(Op opType_, DT dataType_, size_t size_) noexcept
        : Kernel(), dataType(dataType_), opType(opType_), size(size_) {}

    auto K::build(Op op, Tensor const &a, Tensor const &b) noexcept -> KernelBox {
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
        return "Performing binary operation of 2 tensors with same shape on generic cpu";
    }

#define CASE_DT(OP, T)                                                                       \
    case DT::T:                                                                              \
        return [n = this->size](runtime::Resources &, void *workspace, void const *const *inputs, void *const *outputs) { \
            using T_ = primitive<DT::T>::type;                                               \
            auto a = reinterpret_cast<T_ const *>(inputs[0]);                                \
            auto b = reinterpret_cast<T_ const *>(inputs[1]);                                \
            auto c = reinterpret_cast<T_ *>(outputs[0]);                                     \
            std::for_each_n(std::execution::par_unseq,                                       \
                            natural_t(0), n,                                                 \
                            [c, a, b](auto i) { c[i] = OP; });                               \
        }

#define CASE_OP(NAME, LAMBDA)        \
    case Op::NAME:                   \
        switch (dataType.internal) { \
            CASE_DT(LAMBDA, F32);    \
            CASE_DT(LAMBDA, U8);     \
            CASE_DT(LAMBDA, I8);     \
            CASE_DT(LAMBDA, U16);    \
            CASE_DT(LAMBDA, I16);    \
            CASE_DT(LAMBDA, I32);    \
            CASE_DT(LAMBDA, I64);    \
            CASE_DT(LAMBDA, F64);    \
            CASE_DT(LAMBDA, U32);    \
            CASE_DT(LAMBDA, U64);    \
            default:                 \
                UNREACHABLE();       \
        }

    Routine K::lower(Resources &) const noexcept {
        using namespace runtime;

        switch (opType) {
            CASE_OP(Add, a[i] + b[i])
            CASE_OP(Sub, a[i] - b[i])
            CASE_OP(Mul, a[i] * b[i])
            CASE_OP(Div, a[i] / b[i])
            case Op::And:
                switch (dataType.internal) {
                    CASE_DT(a[i] && b[i], Bool);
                    default:
                        UNREACHABLE();
                }
            case Op::Or:
                switch (dataType.internal) {
                    CASE_DT(a[i] || b[i], Bool);
                    default:
                        UNREACHABLE();
                }
            case Op::Xor:
                switch (dataType.internal) {
                    CASE_DT(a[i] ^ b[i], Bool);
                    default:
                        UNREACHABLE();
                }
            case Op::Pow: {
                switch (dataType.internal) {
                    CASE_DT(std::pow(a[i], b[i]), F32);
                    CASE_DT(std::pow(a[i], b[i]), F64);
                    CASE_DT(std::pow(a[i], b[i]), I8);
                    CASE_DT(std::pow(a[i], b[i]), I16);
                    CASE_DT(std::pow(a[i], b[i]), I32);
                    CASE_DT(std::pow(a[i], b[i]), I64);
                    default:
                        UNREACHABLE();
                }
            }
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
