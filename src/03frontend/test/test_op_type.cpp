#include "frontend/operator.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace frontend;

TEST(OpType, parse) {
    OpType::register_("test::op0", nullptr, nullptr);
    OpType::register_("test::op1", nullptr, nullptr);

    auto op0 = OpType::parse("test::op0");
    auto op1 = OpType::parse("test::op1");
    EXPECT_NE(op0, op1);
    auto op_ = OpType::parse("test::op0");
    EXPECT_EQ(op0, op_);
    auto op2 = OpType::tryParse("test::op2");
    EXPECT_FALSE(op2);
}
