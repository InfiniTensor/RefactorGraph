#include "../src/operators/concat.hh"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace frontend;
using namespace onnx;

TEST(infer, Concat) {
    onnx::register_();
    auto edges = Edges{
        {Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(3)}, {}), ""},
        {Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(2)}, {}), ""},
        {Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(5)}, {}), ""},
    };
    count_t inputs[]{0, 1, 2};
    auto infered = Concat(1).infer(TensorRefs(edges, slice(inputs, 3)), {true});
    ASSERT_TRUE(infered.isOk());
    auto outputs = std::move(infered.unwrap());
    ASSERT_EQ(outputs.size(), 1);
    auto y = std::move(outputs[0]);
    ASSERT_EQ(y->dataType, DataType::F32);
    ASSERT_EQ(y->shape, (Shape{DimExpr(2), DimExpr(10)}));
}
