#ifndef COMPUTATION_GATHER_ELEMENTS_H
#define COMPUTATION_GATHER_ELEMENTS_H

#include "../operator.h"

namespace refactor::computation {

    struct GatherElements final : public Operator {
        size_t axis;

        constexpr explicit GatherElements(size_t axis_)
            : Operator(), axis(axis_) {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_GATHER_ELEMENTS_H
