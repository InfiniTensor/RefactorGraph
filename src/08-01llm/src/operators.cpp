#include "llm/operators.h"
#include "operators/rms_normalization.hh"

namespace refactor::llm {
    using namespace frontend;

    void register_() {
        // clang-format off
        #define REGISTER(NAME, CLASS) Operator::register_<CLASS>("llm::" #NAME)
        #undef REGISTER
        // clang-format on
    }

}// namespace refactor::llm
