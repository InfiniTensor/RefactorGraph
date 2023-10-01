#ifndef COMPUTATION_SELECT_H
#define COMPUTATION_SELECT_H

#include "../operator.h"
#include "common/error_handler.h"

namespace refactor::computation {

    enum class SelectType {
        Max,
        Min
    };

    struct Select final : public Operator {
        SelectType type;

        constexpr explicit Select(SelectType type_)
            : Operator(), type(type_) {}

        static size_t typeId(SelectType);
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SELECT_H
