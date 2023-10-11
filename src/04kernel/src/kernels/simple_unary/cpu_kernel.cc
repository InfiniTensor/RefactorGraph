#include "cpu_kernel.hh"
#include "common/error_handler.h"
#include "common/natural.h"
#include <execution>
#include <unordered_set>

namespace refactor::kernel {
    using K = SimpleUnary;
    using Op = SimpleUnaryType;
    using DT = common::DataType;

    K::SimpleUnary(Op opType_, DT dataType_, size_t size_) noexcept
        : Kernel(), dataType(dataType_), opType(opType_), size(size_) {}

    auto K::build(Op op, Tensor const &a) noexcept -> KernelBox {
        static const std::unordered_set<decltype(DT::internal)> TYPE{
            DT::F32, DT::U8, DT::I8, DT::U16, DT::I16,
            DT::I32, DT::I64, DT::F64, DT::U32, DT::U64};
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

#define CASE_DT(OP, T)                                                                       \
    case DT::T:                                                                              \
        return [n = this->size](runtime::Resources &, Addresses inputs, Addresses outputs) { \
            using T_ = common::primitive_t<DT::T>::type;                                     \
            auto x = static_cast<T_ const *>(inputs[0]);                                     \
            auto y = static_cast<T_ *>(outputs[0]);                                          \
            std::for_each_n(std::execution::par_unseq,                                       \
                            common::natural_t(0), n,                                         \
                            [y, x](auto i) { y[i] = OP(x[i]); });                            \
        }

#define CASE_OP(NAME, SYMBOL)        \
    case Op::NAME:                   \
        switch (dataType.internal) { \
            CASE_DT(SYMBOL, F32);    \
            CASE_DT(SYMBOL, U8);     \
            CASE_DT(SYMBOL, I8);     \
            CASE_DT(SYMBOL, U16);    \
            CASE_DT(SYMBOL, I16);    \
            CASE_DT(SYMBOL, I32);    \
            CASE_DT(SYMBOL, I64);    \
            CASE_DT(SYMBOL, F64);    \
            CASE_DT(SYMBOL, U32);    \
            CASE_DT(SYMBOL, U64);    \
            default:                 \
                UNREACHABLE();       \
        }

    auto K::lower() const noexcept -> Operation {
        // clang-format off
        switch (opType) {
            // CASE(Abs    , )
            // CASE(Acos   , )
            // CASE(Acosh  , )
            // CASE(Asin   , )
            // CASE(Asinh  , )
            // CASE(Atan   , )
            // CASE(Atanh  , )
            // CASE(Cos    , )
            // CASE(Cosh   , )
            // CASE(Sin    , )
            // CASE(Sinh   , )
            // CASE(Tan    , )
            // CASE(Tanh   , )
            // CASE(Relu   , )
            // CASE(Sqrt   , )
            // CASE(Sigmoid, )
            // CASE(Erf    , )
            // CASE(Not    , )
            default:
                UNREACHABLE();
        }
        // clang-format on
    }

}// namespace refactor::kernel
