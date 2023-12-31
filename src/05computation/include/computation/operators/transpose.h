﻿#ifndef COMPUTATION_TRANSPOSE_H
#define COMPUTATION_TRANSPOSE_H

#include "../operator.h"
#include "kernel/collectors/transpose.h"
#include <absl/container/inlined_vector.h>

namespace refactor::computation {

    struct Transpose final : public Operator {
        kernel::Permutation perm;

        explicit Transpose(decltype(perm) perm_) noexcept
            : Operator(), perm(std::move(perm_)) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        bool isLayoutDependent() const noexcept final;
        void transposeTo(LayoutType) noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_TRANSPOSE_H
