#ifndef FRONTEND_LOWER_H
#define FRONTEND_LOWER_H

#include "computation/operator.h"
#include "tensor.h"
#include <absl/container/inlined_vector.h>

namespace refactor::frontend {

    struct LowerOperator {
        computation::SharedOp op;
        absl::InlinedVector<size_t, 2> inputs;
    };

    class Operator;
    using LowerFn = LowerOperator (*)(Operator const &, TensorRefs);

    LowerOperator unreachableLower(Operator const &, TensorRefs);

}// namespace refactor::frontend

#endif// FRONTEND_LOWER_H
