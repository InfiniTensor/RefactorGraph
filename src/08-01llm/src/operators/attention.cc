#include "computation/operators/attention.h"
#include "attention.hh"
#include "common.h"

namespace refactor::llm {
    using Op = Attention;

    Op::Attention(decltype(maxSeqLen) maxSeqLen_)
        : Operator(), maxSeqLen(maxSeqLen_) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto maxSeqLen = attributes.getOrInsert("max_seq_len", {0}).float_();
        return OpBox(std::make_unique<Op>(maxSeqLen));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "llm::Attention"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        TODO("");
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        TODO("");
    }

}// namespace refactor::llm
