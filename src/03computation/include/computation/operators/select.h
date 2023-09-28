#ifndef COMPUTATION_SELECT_H
#define COMPUTATION_SELECT_H

#include "../operator.h"

namespace refactor::computation {

    enum class SelectType {
        Max,
        Min
    };

    struct Select : public Operator {
        SelectType type;

        constexpr explicit Select(SelectType type_)
            : Operator(), type(type_) {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_SELECT_H
