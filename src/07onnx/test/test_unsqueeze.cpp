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
    {
        auto &axes = edges[1].tensor;
        int64_t val[]{2, 0};
        std::copy(val, val + 2, reinterpret_cast<int64_t *>(axes->malloc()));
    }
    graph_topo::idx_t inputs[]{0, 1};
    auto infered = Unsqueeze().infer(TensorRefs(edges, slice(inputs, 2)), {true});
    ASSERT_TRUE(infered.isOk());
    auto outputs = std::move(infered.unwrap());
    ASSERT_EQ(outputs.size(), 1);
    auto y = std::move(outputs[0]);
    ASSERT_EQ(y->dataType, DataType::F32);
    ASSERT_EQ(y->shape, (Shape{DimExpr(1), DimExpr(3), DimExpr(1), DimExpr(5)}));
}
