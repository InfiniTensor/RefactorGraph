﻿#ifndef COMPUTATION_IDENTITY_H
#define COMPUTATION_IDENTITY_H

#include "../operator.h"

namespace refactor::computation {

    struct Identity final : public Operator {

        constexpr Identity() noexcept = default;

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        bool isIdentity() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_IDENTITY_H
