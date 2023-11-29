#include "../src/operators/unsqueeze.hh"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace frontend;
using namespace onnx;

TEST(infer, Unsqueeze) {
    onnx::register_();
    auto edges = Edges{
        {Tensor::share(DataType::F32, Shape{DimExpr(3), DimExpr(5)}, {}), ""},
        {Tensor::share(DataType::I64, Shape{DimExpr(2)}, {}), ""},
    };
    int64_t axes[]{2, 0};
    std::memcpy(edges[1].tensor->malloc(), axes, sizeof(axes));
    count_t inputs[]{0, 1};
    auto infered = Unsqueeze(std::nullopt).infer(TensorRefs(edges, inputs), {true});
    ASSERT_TRUE(infered.isOk());
    auto outputs = std::move(infered.unwrap());
    ASSERT_EQ(outputs.size(), 1);
    auto y = std::move(outputs[0]);
    ASSERT_EQ(y->dataType, DataType::F32);
    ASSERT_EQ(y->shape, (Shape{DimExpr(1), DimExpr(3), DimExpr(1), DimExpr(5)}));
}
