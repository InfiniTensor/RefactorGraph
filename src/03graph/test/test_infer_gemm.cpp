#include "../src/infer/infer.h"
#include <gtest/gtest.h>

using namespace refactor::common;
using namespace refactor::graph;

TEST(EdgeInfer, Gemm) {
    {
        auto inputs = Edges{{
            Tensor{DataType::F32, {2, 3}},
            Tensor{DataType::F32, {3, 5}},
        }};
        auto result = inferGemm(inputs, false, false);
        ASSERT_TRUE(result.isOk());
        ASSERT_EQ(result.unwrap(), (Edges{Tensor{DataType::F32, {2, 5}}}));
    }
    {
        auto inputs = Edges{{
            Tensor{DataType::F32, {3, 2}},
            Tensor{DataType::F32, {3, 5}},
        }};
        auto result = inferGemm(inputs, true, false);
        ASSERT_TRUE(result.isOk());
        ASSERT_EQ(result.unwrap(), (Edges{Tensor{DataType::F32, {2, 5}}}));
    }
    {
        auto inputs = Edges{{
            Tensor{DataType::F32, {2, 3}},
            Tensor{DataType::F32, {5, 3}},
        }};
        auto result = inferGemm(inputs, false, true);
        ASSERT_TRUE(result.isOk());
        ASSERT_EQ(result.unwrap(), (Edges{Tensor{DataType::F32, {2, 5}}}));
    }
    {
        auto inputs = Edges{{
            Tensor{DataType::F32, {2, 3}},
            Tensor{DataType::F32, {5, 3}},
            Tensor{DataType::F32, {2, 1}},
        }};
        auto result = inferGemm(inputs, false, true);
        ASSERT_TRUE(result.isOk());
        ASSERT_EQ(result.unwrap(), (Edges{Tensor{DataType::F32, {2, 5}}}));
    }
    {
        auto inputs = Edges{{
            Tensor{DataType::F32, {2, 3}},
            Tensor{DataType::F32, {2, 3}},
        }};
        auto result = inferGemm(inputs, false, false);
        ASSERT_TRUE(result.isErr());
    }
    {
        auto inputs = Edges{{
            Tensor{DataType::F32, {2, 3}},
            Tensor{DataType::F32, {3, 5}},
            Tensor{DataType::F32, {2}},
        }};
        auto result = inferGemm(inputs, false, false);
        ASSERT_TRUE(result.isErr());
    }
}
