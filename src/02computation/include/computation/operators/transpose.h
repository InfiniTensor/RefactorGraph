#ifndef COMPUTATION_TRANSPOSE_H
#define COMPUTATION_TRANSPOSE_H

#include "../operator.h"
#include <absl/container/inlined_vector.h>

namespace refactor::computation {

    struct Transpose : public Operator {
        absl::InlinedVector<size_t, 4> perm;

        explicit Transpose(absl::InlinedVector<size_t, 4> perm_)
            : Operator(), perm(std::move(perm_)) {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_TRANSPOSE_H
