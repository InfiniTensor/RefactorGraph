﻿#include "kernel/attributes/split_info.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, SplitInfo) {
    std::vector<Arc<Tensor>> outputs{
        Tensor::share(DataType::I64, Shape{2, 3, 1, 1, 7, 7}),// 勿
        Tensor::share(DataType::I64, Shape{2, 3, 1, 9, 7, 7}),// 忘
        Tensor::share(DataType::I64, Shape{2, 3, 1, 3, 7, 7}),// 国
        Tensor::share(DataType::I64, Shape{2, 3, 1, 7, 7, 7}),// 耻
    };
    TensorRefs outputs_;
    outputs_.reserve(outputs.size());
    std::transform(outputs.begin(), outputs.end(),
                   std::back_inserter(outputs_),
                   [](auto const &it) { return std::cref(*it); });
    SplitInfo info(3, outputs_);
    EXPECT_EQ(info.blockCount, 6);
    EXPECT_EQ(info.unit(16), 8);
    auto postfix = 49 * 8u;
    EXPECT_EQ(info.sum, 20 * postfix);
    EXPECT_EQ(info.segments, (absl::InlinedVector<dim_t, 4>{1 * postfix, 9 * postfix, 3 * postfix, 7 * postfix}));
    EXPECT_EQ(info.unit(4), 4);
}
