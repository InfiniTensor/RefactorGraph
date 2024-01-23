#include "rms_normalization.hh"
#include "common.h"

namespace refactor::llm {
    using Op = RmsNormalization;

    auto Op::build(ModelContext const &, std::string_view, Attributes) -> OpBox {
        return OpBox(std::make_unique<Op>());
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "llm::RmsNormalization"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(2)

        TODO("");
    }

}// namespace refactor::llm
