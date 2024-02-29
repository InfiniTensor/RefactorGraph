#include "llm/operators.h"
#include "operators/attention.hh"
#include "operators/mat_mul.hh"
#include "operators/rms_normalization.hh"
#include "operators/rope.hh"

namespace refactor::llm {
    using namespace frontend;

    void register_() {
#define REGISTER(NAME, CLASS) Operator::register_<CLASS>("llm::" #NAME)
        // clang-format off
        REGISTER(Attention              , Attention              );
        REGISTER(RmsNormalization       , RmsNormalization       );
        REGISTER(MatMul                 , MatMul                 );
        REGISTER(RotaryPositionEmbedding, RotaryPositionEmbedding);
        // clang-format on
#undef REGISTER
    }

}// namespace refactor::llm
