#include "infer.h"

namespace refactor::communication {
    using namespace computation;

    InferResult inferAllGather(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(1) {
            auto const &input = inputs[0];
            auto nranks = op.attribute("nranks").int_();
            auto ans = Tensors(nranks, nullptr);
            for (auto &t : ans) {
                t = Tensor::share(input->dataType, input->shape);
            }
            return Ok(std::move(ans));
        }
    }
}// namespace refactor::communication
