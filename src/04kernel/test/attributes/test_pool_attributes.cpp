#include "kernel/attributes/pool_attributes.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, PoolAttributes) {
    {
        int64_t const
            dilations[]{1, 2},
            pads[]{3, 4, 5, 6},
            strides[]{7, 8};
        PoolAttributes attr(2, dilations, pads, strides);
        EXPECT_EQ(attr.rank(), 2);
        EXPECT_EQ(attr.dilations()[0], 1);
        EXPECT_EQ(attr.dilations()[1], 2);
        EXPECT_EQ(attr.pads()[0], 3);
        EXPECT_EQ(attr.pads()[1], 4);
        EXPECT_EQ(attr.pads()[2], 5);
        EXPECT_EQ(attr.pads()[3], 6);
        EXPECT_EQ(attr.padsBegin()[0], 3);
        EXPECT_EQ(attr.padsBegin()[1], 4);
        EXPECT_EQ(attr.padsEnd()[0], 5);
        EXPECT_EQ(attr.padsEnd()[1], 6);
        EXPECT_EQ(attr.strides()[0], 7);
        EXPECT_EQ(attr.strides()[1], 8);
    }
    {
        PoolAttributes attr(2, nullptr, nullptr, nullptr);
        EXPECT_EQ(attr.rank(), 2);
        EXPECT_EQ(attr.dilations()[0], 1);
        EXPECT_EQ(attr.dilations()[1], 1);
        EXPECT_EQ(attr.pads()[0], 0);
        EXPECT_EQ(attr.pads()[1], 0);
        EXPECT_EQ(attr.pads()[2], 0);
        EXPECT_EQ(attr.pads()[3], 0);
        EXPECT_EQ(attr.padsBegin()[0], 0);
        EXPECT_EQ(attr.padsBegin()[1], 0);
        EXPECT_EQ(attr.padsEnd()[0], 0);
        EXPECT_EQ(attr.padsEnd()[1], 0);
        EXPECT_EQ(attr.strides()[0], 1);
        EXPECT_EQ(attr.strides()[1], 1);
    }
}
