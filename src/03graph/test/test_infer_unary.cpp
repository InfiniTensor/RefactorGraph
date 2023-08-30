#include "../src/infer/infer.h"
#include <gtest/gtest.h>

using namespace refactor::common;
using namespace refactor::graph;

TEST(EdgeInfer, Unary) {
    {
        auto inputs = Edges{Tensor{DataType::F32, {}}};
        auto result = inferUnary(inputs, isFloatDataType);
        ASSERT_TRUE(result.isOk());
        EXPECT_EQ(result.unwrap(), inputs);
    }
    {
        auto inputs = Edges{{Tensor{DataType::Bool, {}}}};
        auto result = inferUnary(inputs, isBool);
        ASSERT_TRUE(result.isOk());
        EXPECT_EQ(result.unwrap(), inputs);
    }
    {
        auto inputs = Edges{Tensor{DataType::I32, {}}};
        auto result = inferUnary(inputs, isFloatDataType);
        EXPECT_FALSE(result.isOk());
    }
    {
        auto inputs = Edges{{Tensor{DataType::F32, {}}, Tensor{DataType::F32, {}}}};
        auto result = inferUnary(inputs, isFloatDataType);
        EXPECT_FALSE(result.isOk());
    }
}
