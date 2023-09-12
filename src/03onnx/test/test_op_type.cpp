#include "computation/operator.h"
#include "onnx/operators.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace computation;

TEST(OpType, parse) {
    onnx::register_();

    auto add = OpType::parse("onnx::Add");
    auto sub = OpType::parse("onnx::Sub");
    EXPECT_NE(add, sub);
    auto add_ = OpType::parse("onnx::Add");
    EXPECT_EQ(add, add_);
    auto xxx = OpType::tryParse("xxx");
    EXPECT_FALSE(xxx);
}
