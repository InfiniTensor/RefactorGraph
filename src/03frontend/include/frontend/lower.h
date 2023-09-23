#ifndef FRONTEND_LOWER_H
#define FRONTEND_LOWER_H

#include "computation/operator.h"
#include "tensor.h"

namespace refactor::frontend {

    class Operator;
    using LowerFn = computation::SharedOp (*)(Operator const &, Tensors);

    computation::SharedOp unreachableLower(Operator const &, Tensors);

}// namespace refactor::frontend

#endif// FRONTEND_LOWER_H
