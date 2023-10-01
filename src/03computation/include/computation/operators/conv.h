#ifndef COMPUTATION_CONV_H
#define COMPUTATION_CONV_H

#include "../operator.h"

namespace refactor::computation {

    struct Conv final : public Operator {
        constexpr Conv() : Operator() {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_CONV_H
