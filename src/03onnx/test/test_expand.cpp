#include "../src/infer/infer.h"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace common;
using namespace computation;
using namespace onnx;

TEST(infer, Expand) {
    onnx::register_();
    auto opType = OpType::parse("onnx::Expand");
    auto input = Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(3), DimExpr(1)});
    auto shape = Tensor::share(DataType::I64, Shape{DimExpr(4)});
    auto ptr = reinterpret_cast<int64_t *>(shape->malloc());
    ptr[0] = 7;
    ptr[1] = 2;
    ptr[2] = 3;
    ptr[3] = 5;
    auto infered = Operator{opType, {}}.infer({input, shape});
    ASSERT_TRUE(infered.isOk());
    auto outputs = std::move(infered.unwrap());
    ASSERT_EQ(outputs.size(), 1);
    auto output = std::move(outputs[0]);
    ASSERT_EQ(output->dataType, DataType::F32);
    ASSERT_EQ(output->shape, (Shape{DimExpr(7), DimExpr(2), DimExpr(3), DimExpr(5)}));
}
