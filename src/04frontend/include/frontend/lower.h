#ifndef FRONTEND_LOWER_H
#define FRONTEND_LOWER_H

#include "computation/operator.h"
#include <absl/container/inlined_vector.h>

namespace refactor::frontend {

    struct LowerOperator {
        computation::OpBox op;
        absl::InlinedVector<size_t, 2> inputs;
    };

}// namespace refactor::frontend

#endif// FRONTEND_LOWER_H
