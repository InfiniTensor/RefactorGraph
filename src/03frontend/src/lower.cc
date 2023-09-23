#include "frontend/lower.h"
#include "common/error_handler.h"

namespace refactor::frontend {
    computation::SharedOp unreachableLower(Operator const &, Tensors) {
        UNREACHABLE();
    }
}// namespace refactor::frontend
