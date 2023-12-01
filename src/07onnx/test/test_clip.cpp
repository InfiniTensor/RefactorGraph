#include "../src/operators/clip.hh"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace onnx;

TEST(infer, Clip) {
    onnx::register_();
    auto edges = Edges{
        {Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(3)}, {}), ""},
    };
    count_t inputs[]{0};
    auto infered = Clip().infer(TensorRefs(edges, inputs), {true});
    ASSERT_TRUE(infered.isOk());
    auto outputs = std::move(infered.unwrap());
    ASSERT_EQ(outputs.size(), 1);
    auto y = std::move(outputs[0]);
    ASSERT_EQ(y->dataType, DataType::F32);
    ASSERT_EQ(y->shape, (Shape{DimExpr(2), DimExpr(3)}));
}
