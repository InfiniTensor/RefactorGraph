#include "../src/infer/infer.h"
#include <gtest/gtest.h>

using namespace refactor::graph;

TEST(ShapeBroadcast, Unidirectional) {
    EXPECT_TRUE(unidirBroadcast({2, 3, 4, 5}, {}));
    EXPECT_TRUE(unidirBroadcast({2, 3, 4, 5}, {5}));
    EXPECT_TRUE(unidirBroadcast({2, 3, 4, 5}, {2, 1, 1, 5}));
    EXPECT_TRUE(unidirBroadcast({2, 3, 4, 5}, {3, 1, 5}));
}

TEST(ShapeBroadcast, Multidirectional) {
    std::tuple<int, Shape, Shape, Shape> cases[]{
        {0, {2, 3, 4, 5}, {2, 3, 4, 5}, {}},
        {1, {2, 3, 4, 5}, {2, 3, 4, 5}, {5}},
        {2, {2, 3, 4, 5}, {4, 5}, {2, 3, 4, 5}},
        {3, {2, 3, 4, 5}, {1, 4, 5}, {2, 3, 1, 1}},
        {4, {2, 3, 4, 5}, {3, 4, 5}, {2, 1, 1, 1}},
    };
    for (auto const &[idx, ans, a, b] : cases) {
        auto result = multidirBroadcast({a, b});
        ASSERT_TRUE(result.isOk());
        EXPECT_EQ(ans, result.unwrap());
        logi("Multidirectional broadcast test case #{} pass", idx);
    }
}
