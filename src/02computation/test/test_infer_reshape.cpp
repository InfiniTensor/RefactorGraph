#include "../src/infer/infer.h"
#include <gtest/gtest.h>

using namespace refactor::common;
using namespace refactor::graph;

TEST(EdgeInfer, Reshape) {
    {
        auto inputs = Edges{{
            Tensor{DataType::F32, {1, 3, 224, 336}},
            ShapeVariable{{0, 0, 6, 112, -1}},
        }};
        auto result = inferReshape(inputs);
        ASSERT_TRUE(result.isOk());
        ASSERT_EQ(result.unwrap(), (Edges{{Tensor{DataType::F32, {1, 3, 6, 112, 112}}}}));
    }
    {
        auto inputs = Edges{{
            Tensor{DataType::F32, {1, 3, 224, 336}},
            ShapeVariable{{0, 0, 2, 0, -1}},
        }};
        auto result = inferReshape(inputs);
        ASSERT_TRUE(result.isOk());
        ASSERT_EQ(result.unwrap(), (Edges{{Tensor{DataType::F32, {1, 3, 2, 336, 112}}}}));
    }
    {
        auto inputs = Edges{{
            Tensor{DataType::F32, {1, 3, 224, 336}},
            ShapeVariable{{0, 0, 2, 0, 0}},
        }};
        auto result = inferReshape(inputs);
        ASSERT_TRUE(result.isErr());
    }
    {
        auto inputs = Edges{{
            Tensor{DataType::F32, {1, 3, 224, 336}},
            ShapeVariable{{0, 0, 2, -1, -1}},
        }};
        auto result = inferReshape(inputs);
        ASSERT_TRUE(result.isErr());
    }
}
