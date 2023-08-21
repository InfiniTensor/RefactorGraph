#include "../src/infer/infer.h"
#include <gtest/gtest.h>

using namespace refactor::graph;

TEST(ShapeInfer, PoolFunction) {
    {
        auto result = pool({8, 8}, {3, 3}, {{2, 2}}, {{0, 0, 0, 0}}, {{1, 1}});
        EXPECT_TRUE(result.isOk());
        EXPECT_EQ(Shape({4, 4}), result.unwrap());
    }
    {
        auto result = pool({8, 8}, {3, 3}, {{1, 1}}, {{2, 3, 4, 5}}, {});
        EXPECT_TRUE(result.isOk());
        EXPECT_EQ(Shape({12, 14}), result.unwrap());
    }
    {
        auto result = pool({8, 8}, {3, 3}, {}, {}, {});
        EXPECT_TRUE(result.isOk());
        EXPECT_EQ(Shape({6, 6}), result.unwrap());
    }
}
