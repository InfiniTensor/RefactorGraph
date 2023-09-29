#ifndef COMPUTATION_BROADCAST_H
#define COMPUTATION_BROADCAST_H

#include "../operator.h"

namespace refactor::computation {

    struct Broadcast final : public Operator {
        constexpr Broadcast() : Operator() {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
        bool isLayoutDependent() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_BROADCAST_H
