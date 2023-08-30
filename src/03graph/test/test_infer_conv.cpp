#include "../src/infer/infer.h"
#include <gtest/gtest.h>

using namespace refactor::common;
using namespace refactor::graph;

TEST(EdgeInfer, ConvBasic) {
    auto inputs = Edges{{
        Tensor{DataType::F32, {1, 3, 224, 336}},
        Tensor{DataType::F32, {6, 3, 5, 7}},
    }};
    auto result = inferConv(inputs, {}, {}, {});
    ASSERT_TRUE(result.isOk());
    ASSERT_EQ(result.unwrap(), (Edges{{Tensor{DataType::F32, {1, 6, 220, 330}}}}));
}

TEST(EdgeInfer, ConvBias) {
    {
        auto inputs = Edges{{
            Tensor{DataType::F32, {1, 3, 224, 336}},
            Tensor{DataType::F32, {6, 3, 5, 7}},
            Tensor{DataType::F32, {6}},
        }};
        auto result = inferConv(inputs, {}, {}, {});
        ASSERT_TRUE(result.isOk());
        ASSERT_EQ(result.unwrap(), (Edges{{Tensor{DataType::F32, {1, 6, 220, 330}}}}));
    }
    {
        auto inputs = Edges{{
            Tensor{DataType::F32, {1, 3, 224, 336}},
            Tensor{DataType::F32, {6, 3, 5, 7}},
            Tensor{DataType::F32, {}},
        }};
        auto result = inferConv(inputs, {}, {}, {});
        ASSERT_TRUE(result.isErr());
    }
    {
        auto inputs = Edges{{
            Tensor{DataType::F32, {1, 3, 224, 336}},
            Tensor{DataType::F32, {6, 3, 5, 7}},
            Tensor{DataType::F32, {1, 6}},
        }};
        auto result = inferConv(inputs, {}, {}, {});
        ASSERT_TRUE(result.isErr());
    }
}

TEST(EdgeInfer, ConvDilations) {
    auto inputs = Edges{{
        Tensor{DataType::F32, {1, 3, 224, 336}},
        Tensor{DataType::F32, {6, 3, 5, 7}},
    }};
    auto result = inferConv(inputs, {{2, 7}}, {}, {});
    ASSERT_TRUE(result.isOk());
    ASSERT_EQ(result.unwrap(), (Edges{{Tensor{DataType::F32, {1, 6, 216, 294}}}}));
}

TEST(EdgeInfer, ConvPads) {
    auto inputs = Edges{{
        Tensor{DataType::F32, {1, 3, 224, 336}},
        Tensor{DataType::F32, {6, 3, 5, 7}},
    }};
    auto result = inferConv(inputs, {}, {{1, 2, 3, 4}}, {});
    ASSERT_TRUE(result.isOk());
    ASSERT_EQ(result.unwrap(), (Edges{{Tensor{DataType::F32, {1, 6, 224, 336}}}}));
}

TEST(EdgeInfer, ConvStrides) {
    auto inputs = Edges{{
        Tensor{DataType::F32, {1, 3, 224, 336}},
        Tensor{DataType::F32, {6, 3, 5, 7}},
    }};
    auto result = inferConv(inputs, {}, {}, {{2, 3}});
    ASSERT_TRUE(result.isOk());
    ASSERT_EQ(result.unwrap(), (Edges{{Tensor{DataType::F32, {1, 6, 110, 110}}}}));
}

TEST(EdgeInfer, Group) {
    auto inputs = Edges{{
        Tensor{DataType::F32, {1, 6, 224, 336}},
        Tensor{DataType::F32, {6, 3, 5, 7}},
    }};
    auto result = inferConv(inputs, {}, {}, {});
    ASSERT_TRUE(result.isOk());
    ASSERT_EQ(result.unwrap(), (Edges{{Tensor{DataType::F32, {1, 12, 220, 330}}}}));
}
