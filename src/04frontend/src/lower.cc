#include "frontend/lower.h"
#include "common/error_handler.h"

namespace refactor::frontend {

    LowerOperator unreachableLower(Operator const &, TensorRefs) {
        UNREACHABLE();
    }

}// namespace refactor::frontend
