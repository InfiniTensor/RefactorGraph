#include "all_reduce.hh"
#include "common.h"

namespace refactor::communication {
    using Op = AllReduce;

    Op::AllReduce() : Operator() {}

    auto Op::build(ModelContext const &, std::string_view, Attributes) -> OpBox {
        return OpBox(std::make_unique<Op>());
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "AllReduce"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(1)

        return Ok(Tensors{Tensor::share(inputs[0].dataType,
                                        inputs[0].shape,
                                        extractDependency(inputs))});
    }

}// namespace refactor::communication
