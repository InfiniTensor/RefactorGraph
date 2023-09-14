#include "../../src/infer/infer.h"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace common;
using namespace computation;
using namespace onnx;

TEST(infer, CumSum) {
    onnx::register_();
    std::shared_ptr<Tensor> x, axis;
    {
        auto shape = Shape{DimExpr(2), DimExpr(3)};
        auto size = sizeOf(shape);
        auto dataType = DataType::F32;
        auto eleSize = dataTypeSize(dataType);
        auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
        auto ptr = reinterpret_cast<float *>(blob->ptr);
        for (size_t i = 0; i < size; ++i) { ptr[i] = i + 1; }
        x = Tensor::share(dataType, std::move(shape), std::move(blob));
    }
    {
        auto shape = Shape{};
        auto size = sizeOf(shape);
        auto dataType = DataType::I32;
        auto eleSize = dataTypeSize(dataType);
        auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
        auto ptr = reinterpret_cast<int32_t *>(blob->ptr);
        ptr[0] = 0;
        axis = Tensor::share(dataType, std::move(shape), std::move(blob));
    }
    auto op = Operator{OpType::parse("onnx::CumSum"), {}};
    auto infered = inferCumSum(op, {x, axis});
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
