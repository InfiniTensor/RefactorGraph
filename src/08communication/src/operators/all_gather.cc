#include "all_gather.hh"
#include "common.h"

namespace refactor::communication {
    using Op = AllGather;

    Op::AllGather(Int nranks_)
        : Operator(), nranks(nranks_) {}

    auto Op::build(ModelContext const &ctx, std::string_view, Attributes attributes) -> OpBox {
        auto nranks = attributes.at("nranks").int_();
        return OpBox(std::make_unique<Op>(nranks));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "AllReduce"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(1)

        return Ok(Tensors(nranks,
                          Tensor::share(inputs[0].dataType,
                                        inputs[0].shape,
                                        extractDependency(inputs))));
    }

}// namespace refactor::communication
