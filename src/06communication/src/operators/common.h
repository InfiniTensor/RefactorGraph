#ifndef COMMUNICATION_INFER_H
#define COMMUNICATION_INFER_H

#include "common/error_handler.h"
#include "frontend/operator.h"

namespace refactor::communication {
    using namespace frontend;

    // clang-format off
    InferResult inferAllReduce(Operator const &, TensorRefs, InferOptions const&);
    InferResult inferAllGather(Operator const &, TensorRefs, InferOptions const&);
    // clang-format on

#define EXPECT_SIZE(N)                                         \
    if (inputs.size() != (N)) {                                \
        return Err(InferError(ERROR_MSG("Input size error"))); \
    } else

#define EXPECT_VAL(DIM, VAL)                                             \
    int64_t VAL;                                                         \
    if ((DIM).hasValue()) {                                              \
        VAL = (DIM).value();                                             \
    } else {                                                             \
        return Err(InferError(UnknownVariable{(DIM.variable()->name)})); \
    }
}// namespace refactor::communication

#endif// COMMUNICATION_INFER_H
