#include "../src/operators/attention.hh"
#include "llm/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace llm;

TEST(infer, AttentionNoKvCache) {
    llm::register_();
    auto batch = DimExpr("N");
    auto numHead = DimExpr(16);
    auto seqLen = DimExpr(31);
    auto headDim = DimExpr(64);
    {
        auto edges = Edges{
            {Tensor::share(DataType::FP16, Shape{batch, numHead, seqLen, headDim}, {}), ""},
            {Tensor::share(DataType::FP16, Shape{batch, numHead, seqLen, headDim}, {}), ""},
            {Tensor::share(DataType::FP16, Shape{batch, numHead, seqLen, headDim}, {}), ""},
        };
        count_t inputs[]{0, 1, 2};
        auto infered = Attention(0).infer(TensorRefs(edges, inputs), {true});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto y = std::move(outputs[0]);
        ASSERT_EQ(y->dataType, DataType::FP16);
        ASSERT_EQ(y->shape, edges[0].tensor->shape);
    }
    {
        auto edges = Edges{
            {Tensor::share(DataType::FP16, Shape{batch, numHead, seqLen, headDim}, {}), ""},
            {Tensor::share(DataType::FP16, Shape{batch, DimExpr(4), seqLen, headDim}, {}), ""},
            {Tensor::share(DataType::FP16, Shape{batch, DimExpr(4), seqLen, headDim}, {}), ""},
        };
        count_t inputs[]{0, 1, 2};
        auto infered = Attention(0).infer(TensorRefs(edges, inputs), {true});
        ASSERT_TRUE(infered.isOk());
        auto outputs = std::move(infered.unwrap());
        ASSERT_EQ(outputs.size(), 1);
        auto y = std::move(outputs[0]);
        ASSERT_EQ(y->dataType, DataType::FP16);
        ASSERT_EQ(y->shape, edges[0].tensor->shape);
    }
}
