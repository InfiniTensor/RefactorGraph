#include "../src/operators/hard_sigmoid.hh"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace onnx;

TEST(infer, HardSigmoid) {
    onnx::register_();

    auto edges = Edges{
        {Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(3)}, {}), ""},
    };
    count_t inputs[]{0};
    auto infered = HardSigmoid(0.2f, 0.5f).infer(TensorRefs(edges, inputs), {false});
    ASSERT_TRUE(infered.isOk());
    auto outputs = std::move(infered.unwrap());
    ASSERT_EQ(outputs.size(), 1);
    auto y = std::move(outputs[0]);
    ASSERT_EQ(y->dataType, DataType::F32);
    ASSERT_EQ(y->shape, (Shape{DimExpr(2), DimExpr(3)}));
}

