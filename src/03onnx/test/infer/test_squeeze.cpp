#include "../../src/infer/infer.h"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace common;
using namespace computation;
using namespace onnx;

TEST(infer, Squeeze) {
    onnx::register_();
    auto squeeze = OpType::parse("onnx::Squeeze");
    {
        auto x = Tensor::share(DataType::F32, Shape{DimExpr(1), DimExpr(3), DimExpr(1), DimExpr(5)});
        auto infered = Operator{squeeze, {}}.infer({x});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto y = std::move(outputs[0]);
        ASSERT_EQ(y->dataType, DataType::F32);
        ASSERT_EQ(y->shape, (Shape{DimExpr(3), DimExpr(5)}));
    }
    {
        auto x = Tensor::share(DataType::F32, Shape{DimExpr(1), DimExpr(3), DimExpr(1), DimExpr(5)});
        auto axes = Tensor::share(DataType::I64, Shape{DimExpr(1)});
        auto ptr = reinterpret_cast<int64_t *>(axes->malloc());
        ptr[0] = 2;
        auto infered = Operator{squeeze, {}}.infer({x, axes});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto y = std::move(outputs[0]);
        ASSERT_EQ(y->dataType, DataType::F32);
        ASSERT_EQ(y->shape, (Shape{DimExpr(1), DimExpr(3), DimExpr(5)}));
    }
}
