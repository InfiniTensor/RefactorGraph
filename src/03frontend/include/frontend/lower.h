#ifndef FRONTEND_LOWER_H
#define FRONTEND_LOWER_H

#include "computation/operator.h"

namespace refactor::frontend {

    class Operator;
    using LowerFn = computation::SharedOp (*)(Operator const &);

    computation::SharedOp unreachableLower(Operator const &);

}// namespace refactor::frontend

#endif// FRONTEND_LOWER_H
