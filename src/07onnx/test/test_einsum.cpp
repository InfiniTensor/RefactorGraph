#include "../src/operators/einsum.hh"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace frontend;
using namespace onnx;

TEST(infer, Einsum) {
    onnx::register_();
    auto edges = Edges{
        {Tensor::share(DataType::F32, Shape{DimExpr(3), DimExpr(2), DimExpr(5)}, {}), ""},
        {Tensor::share(DataType::F32, Shape{DimExpr(3), DimExpr(5), DimExpr(4)}, {}), ""},
    };
    graph_topo::idx_t inputs[]{0, 1};
    auto infered = Einsum("bik,bkj->bij").infer(TensorRefs(edges, slice(inputs, 2)), {true});
    ASSERT_TRUE(infered.isOk());
    auto outputs = std::move(infered.unwrap());
    ASSERT_EQ(outputs.size(), 1);
    auto y = std::move(outputs[0]);
    ASSERT_EQ(y->dataType, DataType::F32);
    ASSERT_EQ(y->shape, (Shape{DimExpr(3), DimExpr(2), DimExpr(4)}));
}
