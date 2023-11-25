#include "../src/operators/cum_sum.hh"
#include "onnx/operators.h"
#include <gtest/gtest.h>
#include <numeric>

using namespace refactor;
using namespace frontend;
using namespace onnx;

TEST(infer, CumSum) {
    onnx::register_();
    {
        auto edges = Edges{
            {Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(3)}, {}), ""},
            {Tensor::share(DataType::I32, Shape{}, {}), ""},
        };
        auto &x = edges[0].tensor;
        auto ptr = reinterpret_cast<float *>(x->malloc());
        std::iota(ptr, ptr + x->elementsSize(), 1);
        *reinterpret_cast<int32_t *>(edges[1].tensor->malloc()) = 0;
        count_t inputs[]{0, 1};
        auto infered = CumSum(false, false).infer(TensorRefs(edges, inputs), {true});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto y = std::move(outputs[0]);
        ASSERT_EQ(y->dataType, DataType::F32);
        ASSERT_EQ(y->shape, (Shape{DimExpr(2), DimExpr(3)}));
        ASSERT_TRUE(y->data);
        float ans[]{1, 2, 3, 5, 7, 9};
        ASSERT_TRUE(std::equal(ans, ans + y->elementsSize(), y->data->get<float>()));
    }
    {
        auto edges = Edges{
            {Tensor::share(DataType::F32, Shape{DimExpr(2), DimExpr(3)}, {}), ""},
            {Tensor::share(DataType::I32, Shape{}, {}), ""},
        };
        auto &x = edges[0].tensor;
        auto ptr = reinterpret_cast<float *>(x->malloc());
        std::iota(ptr, ptr + x->elementsSize(), 1);
        *reinterpret_cast<int32_t *>(edges[1].tensor->malloc()) = 1;
        count_t inputs[]{0, 1};
        auto infered = CumSum(false, false).infer(TensorRefs(edges, inputs), {true});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto y = std::move(outputs[0]);
        ASSERT_EQ(y->dataType, DataType::F32);
        ASSERT_EQ(y->shape, (Shape{DimExpr(2), DimExpr(3)}));
        ASSERT_TRUE(y->data);
        float ans[]{1, 3, 6, 4, 9, 15};
        ASSERT_TRUE(std::equal(ans, ans + y->elementsSize(), y->data->get<float>()));
    }
}
