#include "../src/infer/infer.h"
#include <gtest/gtest.h>

using namespace refactor::common;
using namespace refactor::graph;

TEST(EdgeInfer, Pool) {
    auto inputs = Edges{{Tensor{DataType::F32, {1, 3, 224, 336}}}};
    auto result = inferPool(inputs, {}, {5, 7}, {}, {});
    ASSERT_TRUE(result.isOk());
    ASSERT_EQ(result.unwrap(), (Edges{{Tensor{DataType::F32, {1, 3, 220, 330}}}}));
}

TEST(EdgeInfer, PoolDilations) {
    auto inputs = Edges{{Tensor{DataType::F32, {1, 3, 224, 336}}}};
    auto result = inferPool(inputs, {{5, 6}}, {5, 7}, {}, {});
    ASSERT_TRUE(result.isOk());
    ASSERT_EQ(result.unwrap(), (Edges{{Tensor{DataType::F32, {1, 3, 204, 300}}}}));
}

TEST(EdgeInfer, PoolPads) {
    auto inputs = Edges{{Tensor{DataType::F32, {1, 3, 224, 336}}}};
    auto result = inferPool(inputs, {}, {5, 7}, {{2, 3, 4, 5}}, {});
    ASSERT_TRUE(result.isOk());
    ASSERT_EQ(result.unwrap(), (Edges{{Tensor{DataType::F32, {1, 3, 226, 338}}}}));
}

TEST(EdgeInfer, PoolStrides) {
    auto inputs = Edges{{Tensor{DataType::F32, {1, 3, 224, 336}}}};
    auto result = inferPool(inputs, {}, {5, 7}, {}, {{2, 3}});
    ASSERT_TRUE(result.isOk());
    ASSERT_EQ(result.unwrap(), (Edges{{Tensor{DataType::F32, {1, 3, 110, 110}}}}));
}
