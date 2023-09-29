#ifndef COMPUTATION_SPLIT_H
#define COMPUTATION_SPLIT_H

#include "../operator.h"

namespace refactor::computation {

    struct Split final : public Operator {
        size_t axis;

        constexpr Split(size_t axis_) : Operator(), axis(axis_) {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SPLIT_H
