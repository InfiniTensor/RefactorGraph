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

        constexpr explicit SimpleUnary(SimpleUnaryType type_) noexcept
            : Operator(), type(type_) {}

        static size_t typeId(SimpleUnaryType) noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SIMPLE_UNARY_H
