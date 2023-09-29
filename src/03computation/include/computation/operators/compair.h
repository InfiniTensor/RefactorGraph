#ifndef COMPUTATION_COMPAIR_H
#define COMPUTATION_COMPAIR_H

#include "../operator.h"
#include "common/error_handler.h"

namespace refactor::computation {

    enum class CompairType {
        EQ,
        NE,
        LT,
        LE,
        GT,
        GE,
    };

    struct Compair final : public Operator {
        CompairType type;

        constexpr explicit Compair(CompairType type_)
            : Operator(), type(type_) {}

        static size_t typeId(CompairType);
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_COMPAIR_H
