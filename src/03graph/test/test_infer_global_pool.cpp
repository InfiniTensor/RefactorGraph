#include "../src/infer/infer.h"
#include <gtest/gtest.h>

using namespace refactor::common;
using namespace refactor::graph;

TEST(EdgeInfer, GlobalPool) {
    auto inputs = Edges{{Tensor{DataType::F32, {1, 3, 224, 336}}}};
    auto result = inferGlobalPool(inputs);
    ASSERT_TRUE(result.isOk());
    ASSERT_EQ(result.unwrap(), (Edges{{Tensor{DataType::F32, {1, 3, 1, 1}}}}));
}
