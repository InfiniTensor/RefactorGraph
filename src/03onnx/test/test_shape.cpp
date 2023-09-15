#include "../src/infer/infer.h"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace common;
using namespace computation;
using namespace onnx;

TEST(infer, Shape) {
    onnx::register_();
    auto opType = OpType::parse("onnx::Shape");
    auto data = Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(3), DimExpr(1)});
    auto infered = Operator{opType, {}}.infer({data});
    ASSERT_TRUE(infered.isOk());
    auto outputs = std::move(infered.unwrap());
    ASSERT_EQ(outputs.size(), 1);
    auto shape = std::move(outputs[0]);
    ASSERT_EQ(shape->dataType, DataType::I64);
    ASSERT_EQ(shape->shape, (Shape{DimExpr(3)}));
    auto ptr = reinterpret_cast<int64_t *>(shape->data->ptr);
    ASSERT_EQ(ptr[0], 2);
    ASSERT_EQ(ptr[1], 3);
    ASSERT_EQ(ptr[2], 1);
}
