#ifndef COMPUTATION_SIMPLE_BINARY_H
#define COMPUTATION_SIMPLE_BINARY_H

#include "../operator.h"
#include "common/error_handler.h"

namespace refactor::computation {

    enum class SimpleBinaryType {
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        And,
        Or,
        Xor,
    };

    struct SimpleBinary final : public Operator {
        SimpleBinaryType type;

        constexpr explicit SimpleBinary(SimpleBinaryType type_)
            : Operator(), type(type_) {}

        static size_t typeId(SimpleBinaryType);
        size_t opTypeId() const override;
        std::string_view name() const override;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SIMPLE_BINARY_H
