#include "operators.h"
#include "operators/moe.hh"

namespace refactor::moe {
    using namespace frontend;

    void register_() {
#define REGISTER(NAME, CLASS) Operator::register_<CLASS>("moe::" #NAME)
        // clang-format off
        REGISTER(AssignPos       , AssignPos       );
        REGISTER(Reorder         , Reorder        );
        // clang-format on
#undef REGISTER
    }

}// namespace refactor::moe
