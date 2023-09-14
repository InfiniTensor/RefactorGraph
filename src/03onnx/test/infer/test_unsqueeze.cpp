#include "../../src/infer/infer.h"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace common;
using namespace computation;
using namespace onnx;

TEST(infer, Unsqueeze) {
    onnx::register_();
    auto unsqueeze = OpType::parse("onnx::Unsqueeze");
    {
        auto x = Tensor::share(DataType::F32, Shape{DimExpr(3), DimExpr(5)});
        auto axes = Tensor::share(DataType::I64, Shape{DimExpr(2)});
        auto ptr = reinterpret_cast<int64_t *>(axes->malloc());
        ptr[0] = 2;
        ptr[1] = 0;
        auto infered = Operator{unsqueeze, {}}.infer({x, axes});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto y = std::move(outputs[0]);
        ASSERT_EQ(y->dataType, DataType::F32);
        ASSERT_EQ(y->shape, (Shape{DimExpr(1), DimExpr(3), DimExpr(1), DimExpr(5)}));
    }
}
