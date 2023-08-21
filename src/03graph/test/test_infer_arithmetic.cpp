#include "../src/infer/infer.h"
#include <gtest/gtest.h>

using namespace refactor::common;
using namespace refactor::graph;

TEST(EdgeInfer, Arithmetic) {
    {
        auto inputs = Edges{{Tensor{DataType::I32, {1, 1, 3}},
                             Tensor{DataType::I32, {2, 3}}}};
        auto result = inferArithmetic(inputs);
        ASSERT_TRUE(result.isOk());
        ASSERT_EQ(result.unwrap(), (Edges{Tensor{DataType::I32, {1, 2, 3}}}));
    }
    {
        auto inputs = Edges{{Tensor{DataType::I32, {1, 1, 3}},
                             Tensor{DataType::I32, {1, 2}}}};
        auto result = inferArithmetic(inputs);
        ASSERT_FALSE(result.isOk());
    }
    {
        auto inputs = Edges{{Tensor{DataType::I32, {1, 1, 3}},
                             Tensor{DataType::F32, {2, 3}}}};
        auto result = inferArithmetic(inputs);
        ASSERT_FALSE(result.isOk());
    }
    {
        auto inputs = Edges{{Tensor{DataType::Bool, {1, 1, 3}},
                             Tensor{DataType::Bool, {2, 3}}}};
        auto result = inferArithmetic(inputs);
        ASSERT_FALSE(result.isOk());
    }
    {
        auto inputs = Edges{{Tensor{DataType::I32, {1, 1, 3}}}};
        auto result = inferArithmetic(inputs);
        ASSERT_FALSE(result.isOk());
    }
}
