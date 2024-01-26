#include "../src/operators/rms_normalization.hh"
#include "llm/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace llm;

TEST(infer, RmsNormalization) {
    llm::register_();
    auto edges = Edges{
        {Tensor::share(DataType::F32, Shape{DimExpr(7), DimExpr(2), DimExpr(3)}, {}), ""},
        {Tensor::share(DataType::F32, Shape{DimExpr(3)}, {}), ""},
    };
    count_t inputs[]{0, 1};
    auto infered = RmsNormalization(1e-6).infer(TensorRefs(edges, inputs), {true});
    ASSERT_TRUE(infered.isOk());
    auto outputs = std::move(infered.unwrap());
    ASSERT_EQ(outputs.size(), 1);
    auto y = std::move(outputs[0]);
    ASSERT_EQ(y->dataType, DataType::F32);
    ASSERT_EQ(y->shape, (Shape{DimExpr(7), DimExpr(2), DimExpr(3)}));
}
