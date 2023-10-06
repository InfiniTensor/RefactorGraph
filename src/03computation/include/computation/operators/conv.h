#ifndef COMPUTATION_CONV_H
#define COMPUTATION_CONV_H

#include "../operator.h"

namespace refactor::computation {

    struct Conv final : public Operator {
        constexpr Conv() noexcept : Operator() {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_CONV_H
