#include "common.h"
#include <gtest/gtest.h>

using namespace refactor;

TEST(graph_topo, Builder) {
    float val = 2047;
    fp16_t ans(val);
    EXPECT_EQ(ans.to_f32(), val);
    fmt::println("{}", ans.to_string());
}
