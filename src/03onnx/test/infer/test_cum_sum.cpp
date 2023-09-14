#include "../../src/infer/infer.h"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace common;
using namespace computation;
using namespace onnx;

TEST(infer, CumSum) {
    onnx::register_();
    auto cumSum = OpType::parse("onnx::CumSum");
    std::shared_ptr<Tensor> x, axis;
    {
        {
            x = Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(3)});
            auto ptr = reinterpret_cast<float *>(x->malloc());
            auto size = x->elementsSize();
            for (size_t i = 0; i < size; ++i) { ptr[i] = i + 1; }
        }
        {
            axis = Tensor::share(DataType::I32, Shape{});
            auto ptr = reinterpret_cast<int32_t *>(axis->malloc());
            ptr[0] = 0;
        }
        auto infered = Operator{cumSum, {}}.infer({x, axis});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto y = std::move(outputs[0]);
        ASSERT_EQ(y->dataType, DataType::F32);
        ASSERT_EQ(y->shape, (Shape{DimExpr(2), DimExpr(3)}));
        auto ptr = reinterpret_cast<float *>(y->data->ptr);
        ASSERT_EQ(ptr[0], 1.0);
        ASSERT_EQ(ptr[1], 2.0);
        ASSERT_EQ(ptr[2], 3.0);
        ASSERT_EQ(ptr[3], 5.0);
        ASSERT_EQ(ptr[4], 7.0);
        ASSERT_EQ(ptr[5], 9.0);
    }
    {
        {
            x = Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(3)});
            auto ptr = reinterpret_cast<float *>(x->malloc());
            auto size = x->elementsSize();
            for (size_t i = 0; i < size; ++i) { ptr[i] = i + 1; }
        }
        {
            axis = Tensor::share(DataType::I32, Shape{});
            auto ptr = reinterpret_cast<int32_t *>(axis->malloc());
            ptr[0] = 1;
        }
        auto infered = Operator{cumSum, {}}.infer({x, axis});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto y = std::move(outputs[0]);
        ASSERT_EQ(y->dataType, DataType::F32);
        ASSERT_EQ(y->shape, (Shape{DimExpr(2), DimExpr(3)}));
        auto ptr = reinterpret_cast<float *>(y->data->ptr);
        ASSERT_EQ(ptr[0], 1.0);
        ASSERT_EQ(ptr[1], 3.0);
        ASSERT_EQ(ptr[2], 6.0);
        ASSERT_EQ(ptr[3], 4.0);
        ASSERT_EQ(ptr[4], 9.0);
        ASSERT_EQ(ptr[5], 15.0);
    }
}
