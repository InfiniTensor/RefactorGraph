#include "../src/operators/shape.hh"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace frontend;
using namespace onnx;

TEST(infer, Shape) {
    onnx::register_();
    {
        auto edges = Edges{{Tensor::share(DataType::F32, frontend::Shape{DimExpr(2), DimExpr(3), DimExpr(1)}, {}), ""}};
        count_t inputs[]{0};
        auto infered = onnx::Shape(0, std::nullopt).infer(TensorRefs(edges, slice(inputs, 1)), {true});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto shape = std::move(outputs[0]);
        ASSERT_EQ(shape->dataType, DataType::I64);
        ASSERT_EQ(shape->shape, (frontend::Shape{DimExpr(3)}));
        ASSERT_TRUE(shape->data);
        int64_t ans[]{2, 3, 1};
        ASSERT_TRUE(std::equal(ans, ans + shape->elementsSize(), shape->data->get<int64_t>()));
    }
    {
        auto edges = Edges{{Tensor::share(DataType::F32, frontend::Shape{DimExpr(2), DimExpr(3), DimExpr(1)}, {}), ""}};
        count_t inputs[]{0};
        auto infered = onnx::Shape(1, std::nullopt).infer(TensorRefs(edges, slice(inputs, 1)), {true});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto shape = std::move(outputs[0]);
        ASSERT_EQ(shape->dataType, DataType::I64);
        ASSERT_EQ(shape->shape, (frontend::Shape{DimExpr(2)}));
        ASSERT_TRUE(shape->data);
        int64_t ans[]{3, 1};
        ASSERT_TRUE(std::equal(ans, ans + shape->elementsSize(), shape->data->get<int64_t>()));
    }
    {
        auto edges = Edges{{Tensor::share(DataType::F32, frontend::Shape{DimExpr(2), DimExpr(3), DimExpr(1)}, {}), ""}};
        count_t inputs[]{0};
        auto infered = onnx::Shape(1, {2}).infer(TensorRefs(edges, slice(inputs, 1)), {true});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto shape = std::move(outputs[0]);
        ASSERT_EQ(shape->dataType, DataType::I64);
        ASSERT_EQ(shape->shape, (frontend::Shape{DimExpr(1)}));
        ASSERT_TRUE(shape->data);
        int64_t ans[]{3};
        ASSERT_TRUE(std::equal(ans, ans + shape->elementsSize(), shape->data->get<int64_t>()));
    }
    {
        auto edges = Edges{{Tensor::share(DataType::F32, frontend::Shape{DimExpr(2), DimExpr(3), DimExpr(1)}, {}), ""}};
        count_t inputs[]{0};
        auto infered = onnx::Shape(1, {-1}).infer(TensorRefs(edges, slice(inputs, 1)), {true});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto shape = std::move(outputs[0]);
        ASSERT_EQ(shape->dataType, DataType::I64);
        ASSERT_EQ(shape->shape, (frontend::Shape{DimExpr(1)}));
        ASSERT_TRUE(shape->data);
        int64_t ans[]{3};
        ASSERT_TRUE(std::equal(ans, ans + shape->elementsSize(), shape->data->get<int64_t>()));
    }
}
