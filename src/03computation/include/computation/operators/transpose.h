#ifndef COMPUTATION_TRANSPOSE_H
#define COMPUTATION_TRANSPOSE_H

#include "../operator.h"
#include <absl/container/inlined_vector.h>

namespace refactor::computation {

    struct Transpose final : public Operator {
        absl::InlinedVector<size_t, 4> perm;

        explicit Transpose(absl::InlinedVector<size_t, 4> perm_)
            : Operator(), perm(std::move(perm_)) {}

        static size_t typeId();
        size_t opTypeId() const final;
        std::string_view name() const final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_TRANSPOSE_H
