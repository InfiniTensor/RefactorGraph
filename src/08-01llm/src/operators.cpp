#include "llm/operators.h"
#include "operators/mat_mul.hh"
#include "operators/rms_normalization.hh"

namespace refactor::llm {
    using namespace frontend;

    void register_() {
#define REGISTER(NAME, CLASS) Operator::register_<CLASS>("llm::" #NAME)
        // clang-format off
        REGISTER(MatMul          , MatMul          );
        REGISTER(RmsNormalization, RmsNormalization);
        // clang-format on
#undef REGISTER
    }

}// namespace refactor::llm
