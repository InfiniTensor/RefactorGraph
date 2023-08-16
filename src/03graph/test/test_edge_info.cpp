#include "graph/edge_info.h"
#include <gtest/gtest.h>

using namespace refactor::graph;
using namespace refactor::common;

TEST(EdgeInfo, EmptyEdgeInfo) {
    EXPECT_FALSE(EmptyEdgeInfo{} == EmptyEdgeInfo{});
}

TEST(EdgeInfo, Tensor) {
    Tensor
        a{DataType::F32, {1, 3, 224, 224}},
        b{DataType::F32, {1, 3, 224, 224}},
        c{DataType::I32, {1, 3, 224, 224}},
        d{DataType::F32, {1, 3, 555, 777}};

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(a, d);
}

TEST(EdgeInfo, ShapeVariable) {
    ShapeVariable
        a{{1, 3, 224, 224}},
        b{{1, 3, 224, 224}},
        c{{1, 3, 555, 777}};

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(EdgeInfo, EdgeInfo) {
    EdgeInfo
        a(Tensor{DataType::F32, {1, 3, 224, 224}}),
        b(Tensor{DataType::F32, {1, 3, 224, 224}}),
        c,
        d(ShapeVariable{{1, 3, 224, 224}});

    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(a, d);
    EXPECT_NE(c, d);

    EXPECT_FALSE(a.isEmpty());
    EXPECT_TRUE(a.isTensor());
    EXPECT_FALSE(a.isShapeVariable());

    EXPECT_TRUE(c.isEmpty());
    EXPECT_FALSE(c.isTensor());
    EXPECT_FALSE(c.isShapeVariable());

    EXPECT_FALSE(d.isEmpty());
    EXPECT_FALSE(d.isTensor());
    EXPECT_TRUE(d.isShapeVariable());
}
