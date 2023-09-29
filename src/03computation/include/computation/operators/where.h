#ifndef COMPUTATION_WHERE_H
#define COMPUTATION_WHERE_H

#include "../operator.h"

namespace refactor::computation {

    struct Where final : public Operator {
        constexpr Where() : Operator() {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_WHERE_H
