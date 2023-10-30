#ifndef COMPUTATION_RESHAPE_H
#define COMPUTATION_RESHAPE_H

#include "../operator.h"

namespace refactor::computation {

    struct Reshape final : public LayoutDependentOperator {
        constexpr Reshape() noexcept : LayoutDependentOperator() {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        bool isIdentity() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_RESHAPE_H
