#include "../src/infer/infer.h"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace common;
using namespace computation;
using namespace onnx;

TEST(infer, Concat) {
    onnx::register_();
    auto opType = OpType::parse("onnx::Concat");
    auto a = Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(3)});
    auto b = Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(2)});
    auto c = Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(5)});
    auto infered = Operator{opType, {{"axis", {1}}}}.infer({a, b, c});
    ASSERT_TRUE(infered.isOk());
    auto outputs = std::move(infered.unwrap());
    ASSERT_EQ(outputs.size(), 1);
    auto y = std::move(outputs[0]);
    ASSERT_EQ(y->dataType, DataType::F32);
    ASSERT_EQ(y->shape, (Shape{DimExpr(2), DimExpr(10)}));
}
