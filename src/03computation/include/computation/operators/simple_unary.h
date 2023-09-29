#ifndef COMPUTATION_SIMPLE_UNARY_H
#define COMPUTATION_SIMPLE_UNARY_H

#include "../operator.h"
#include "common/error_handler.h"

namespace refactor::computation {

    enum class SimpleUnaryType {
        Abs,
        Acos,
        Acosh,
        Asin,
        Asinh,
        Atan,
        Atanh,
        Cos,
        Cosh,
        Sin,
        Sinh,
        Tan,
        Tanh,
        Relu,
        Sqrt,
        Sigmoid,
        Erf,
        Not,
    };

    struct SimpleUnary final : public Operator {
        SimpleUnaryType type;

        constexpr explicit SimpleUnary(SimpleUnaryType type_)
            : Operator(), type(type_) {}

        static size_t typeId(SimpleUnaryType);
        size_t opTypeId() const override;
        std::string_view name() const override;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SIMPLE_UNARY_H
