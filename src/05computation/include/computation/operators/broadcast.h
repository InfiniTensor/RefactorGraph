#ifndef COMPUTATION_BROADCAST_H
#define COMPUTATION_BROADCAST_H

#include "../operator.h"

namespace refactor::computation {

    struct Broadcast final : public LayoutDependentOperator {
        constexpr Broadcast() noexcept : LayoutDependentOperator() {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        bool isIdentity() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_BROADCAST_H
