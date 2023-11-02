#include "basic_cpu.hh"
#include <execution>

namespace refactor::kernel {
    using K = BinaryBasicCpu;
    using Op = SimpleBinaryType;
    using DT = DataType;

    K::BinaryBasicCpu(Op opType_, DT dataType_, Broadcaster b) noexcept
        : Kernel(),
          dataType(dataType_),
          opType(opType_),
          broadcaster(std::move(b)) {}

    auto K::build(Op op, Tensor const &a, Tensor const &b) noexcept -> KernelBox {
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
        return "Performing binary operation of 2 tensors on generic cpu";
    }

#define CASE_DT(OP, T)                                                                                        \
    case DT::T:                                                                                               \
        return [broadcaster = this->broadcaster](runtime::Resources &, void const **inputs, void **outputs) { \
            using T_ = primitive<DT::T>::type;                                                              \
            auto aa = reinterpret_cast<T_ const *>(inputs[0]);                                                \
            auto bb = reinterpret_cast<T_ const *>(inputs[1]);                                                \
            auto cc = reinterpret_cast<T_ *>(outputs[0]);                                                     \
            uint_lv2 ii[2];                                                                                   \
            for (auto i : range0_(broadcaster.outputsCount)) {                                                \
                broadcaster.locate(i, ii);                                                                    \
                auto a = aa[ii[0]], b = bb[ii[1]];                                                            \
                cc[i] = (OP);                                                                                 \
            }                                                                                                 \
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

    auto K::lower() const noexcept -> Routine {
        using namespace runtime;

        switch (opType) {
            CASE_OP(Add, a + b)
            CASE_OP(Sub, a - b)
            CASE_OP(Mul, a * b)
            CASE_OP(Div, a / b)
            case Op::And:
                switch (dataType.internal) {
                    CASE_DT(a && b, Bool);
                    default:
                        UNREACHABLE();
                }
            case Op::Or:
                switch (dataType.internal) {
                    CASE_DT(a || b, Bool);
                    default:
                        UNREACHABLE();
                }
            case Op::Xor:
                switch (dataType.internal) {
                    CASE_DT(a ^ b, Bool);
                    default:
                        UNREACHABLE();
                }
            case Op::Pow: {
                switch (dataType.internal) {
                    CASE_DT(std::pow(a, b), F32);
                    CASE_DT(std::pow(a, b), F64);
                    CASE_DT(std::pow(a, b), I8);
                    CASE_DT(std::pow(a, b), I16);
                    CASE_DT(std::pow(a, b), I32);
                    CASE_DT(std::pow(a, b), I64);
                    default:
                        UNREACHABLE();
                }
            }
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
