#include "../src/operators/pad.hh"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace onnx;

TEST(infer, Pad) {
    onnx::register_();

    {
        auto edges = Edges{
            {Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(3), DimExpr(4)}, {}), ""},
            {Tensor::share(DataType::I64, Shape{DimExpr(6)}, {}), ""},
        };
        auto ptr = reinterpret_cast<int64_t *>(edges[1].tensor->malloc());
        std::fill(ptr, ptr + edges[1].tensor->elementsSize(), 1);
        count_t inputs[]{0, 1};
        auto infered = Pad(PadMode::Constant).infer(TensorRefs(edges, inputs), {false});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto y = std::move(outputs[0]);
        ASSERT_EQ(y->dataType, DataType::F32);
        ASSERT_EQ(y->shape, (Shape{DimExpr(4), DimExpr(5), DimExpr(6)}));
    }
    {
        auto edges = Edges{
            {Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(3), DimExpr(4)}, {}), ""},
            {Tensor::share(DataType::I64, Shape{DimExpr(2)}, {}), ""},
            {Tensor::share(DataType::F32, Shape{DimExpr(1)}, {}), ""},
            {Tensor::share(DataType::I32, Shape{DimExpr(1)}, {}), ""},
        };
        auto ptr_pad = reinterpret_cast<int64_t *>(edges[1].tensor->malloc());
        std::fill(ptr_pad, ptr_pad + edges[1].tensor->elementsSize(), 1);
        auto ptr_axes = reinterpret_cast<int64_t *>(edges[3].tensor->malloc());
        std::fill(ptr_axes, ptr_axes + edges[3].tensor->elementsSize(), 1);
        count_t inputs[]{0, 1, 2, 3};
        auto infered = Pad(PadMode::Constant).infer(TensorRefs(edges, inputs), {false});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto y = std::move(outputs[0]);
        ASSERT_EQ(y->dataType, DataType::F32);
        ASSERT_EQ(y->shape, (Shape{DimExpr(2), DimExpr(5), DimExpr(4)}));
    }
}
