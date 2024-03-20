#include "../src/operators/moe.hh"
#include "operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace moe;

TEST(infer, AssignPos) {
    moe::register_();
    auto edges = Edges{
        
        {Tensor::share(DataType::I16, Shape{DimExpr(8), DimExpr(2)}, {}), ""},//gate 8*2
    };
    count_t inputs[]{0};
    auto infered = AssignPos(2,4).infer(TensorRefs(edges, inputs), {true});
    ASSERT_TRUE(infered.isOk());
    auto outputs = std::move(infered.unwrap());
    ASSERT_EQ(outputs.size(), 2);
    auto expert_cnt = std::move(outputs[0]);
    ASSERT_EQ(expert_cnt->dataType, DataType::F32);
    ASSERT_EQ(expert_cnt->shape, (Shape{DimExpr(4)}));
    auto pos = std::move(outputs[1]);
    ASSERT_EQ(pos->dataType, DataType::I16);
    ASSERT_EQ(pos->shape, (Shape{DimExpr(16)}));
}
