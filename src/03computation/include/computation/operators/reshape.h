#ifndef COMPUTATION_RESHAPE_H
#define COMPUTATION_RESHAPE_H

#include "../operator.h"

namespace refactor::computation {

    struct Reshape final : public Operator {
        constexpr Reshape() : Operator() {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
        bool isLayoutDependent() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_RESHAPE_H
