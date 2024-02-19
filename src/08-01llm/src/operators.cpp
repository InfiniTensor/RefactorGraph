#include "llm/operators.h"
#include "operators/attention.hh"
#include "operators/mat_mul.hh"
#include "operators/rms_normalization.hh"

namespace refactor::llm {
    using namespace frontend;

    void register_() {
#define REGISTER(NAME, CLASS) Operator::register_<CLASS>("llm::" #NAME)
        // clang-format off
        REGISTER(Attention       , Attention       );
        REGISTER(RmsNormalization, RmsNormalization);
        REGISTER(MatMul          , MatMul          );
        // clang-format on
#undef REGISTER
    }

}// namespace refactor::llm
