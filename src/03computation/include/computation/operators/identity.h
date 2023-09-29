#ifndef COMPUTATION_IDENTITY_H
#define COMPUTATION_IDENTITY_H

#include "../operator.h"

namespace refactor::computation {

    struct Identity final : public Operator {
        constexpr Identity() : Operator() {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_IDENTITY_H
