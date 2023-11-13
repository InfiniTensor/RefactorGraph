#include "../src/operators/conv.hh"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace frontend;
using namespace onnx;

TEST(infer, Conv) {
    onnx::register_();
    auto edges = Edges{
        {Tensor::share(DataType::F32, Shape{DimExpr(1), DimExpr(3), DimExpr(224), DimExpr(224)}, {}), ""},
        {Tensor::share(DataType::F32, Shape{DimExpr(64), DimExpr(3), DimExpr(7), DimExpr(7)}, {}), ""},
    };
    count_t inputs[]{0, 1};
    auto infered = Conv({}, {{3, 3, 3, 3}}, {{2, 2}}).infer(TensorRefs(edges, slice(inputs, 2)), {true});
    if (infered.isErr()) { throw infered.unwrapErr(); }
    auto outputs = std::move(infered.unwrap());
    ASSERT_EQ(outputs.size(), 1);
    auto y = std::move(outputs[0]);
    ASSERT_EQ(y->dataType, DataType::F32);
    fmt::println("{}", shapeFormat(y->shape));
    ASSERT_EQ(y->shape, (Shape{DimExpr(1), DimExpr(64), DimExpr(112), DimExpr(112)}));
}
