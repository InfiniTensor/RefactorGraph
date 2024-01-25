#include "llm/operators.h"
#include "operators/mat_mul.hh"

namespace refactor::llm {
    using namespace frontend;

    void register_() {
        // clang-format off
        #define REGISTER(NAME, CLASS) Operator::register_<CLASS>("llm::" #NAME)
        REGISTER(MatMul, MatMul);
        #undef REGISTER
        // clang-format on
    }

}// namespace refactor::llm
