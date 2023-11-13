#include "../src/operators/expand.hh"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace frontend;
using namespace onnx;

TEST(infer, Expand) {
    onnx::register_();
    auto edges = Edges{
        {Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(3), DimExpr(1)}, {}), ""},
        {Tensor::share(DataType::I64, Shape{DimExpr(4)}, {}), ""},
    };
    {
        auto &shape = edges[1].tensor;
        int64_t shape_[]{7, 2, 3, 5};
        std::copy(shape_, shape_ + 4, reinterpret_cast<int64_t *>(shape->malloc()));
    }
    count_t inputs[]{0, 1};
    auto infered = Expand().infer(TensorRefs(edges, slice(inputs, 2)), {true});
    ASSERT_TRUE(infered.isOk());
    auto outputs = std::move(infered.unwrap());
    ASSERT_EQ(outputs.size(), 1);
    auto output = std::move(outputs[0]);
    ASSERT_EQ(output->dataType, DataType::F32);
    ASSERT_EQ(output->shape, (Shape{DimExpr(7), DimExpr(2), DimExpr(3), DimExpr(5)}));
}
