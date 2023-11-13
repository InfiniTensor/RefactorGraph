#include "kernel/attributes/slice_info.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, SliceInfo) {
    auto input = Tensor::share(DataType::F32, Shape{7, 6, 5, 1, 2, 3});
    Dimensions dims{
        {5, -2, 3},// 7 -> {5, 3, 1}    -> {144, 900, -360}
        {2, 3, 2}, // 6 -> {2, 5}       -> { 48,  60,   90}
        {1, 1, 4}, // 5 -> {1, 2, 3, 4} -> { 24,   6,   30}
        {0, 1, 1}, // 1 -> {0}
        {0, 1, 2}, // 2 -> {0, 1}
        {0, 1, 3}, // 3 -> {0, 1, 2}
    };
    SliceInfo info(dims, *input);
    EXPECT_EQ(info.blockCount, 6);
    EXPECT_EQ(info.blockSize, 96);
    EXPECT_EQ(info.baseOffset, 24);
    EXPECT_EQ(info.dims,
              // clang-format off
              (decltype(info.dims){
                  {48 / 24, 900 * 4, -360 * 4},
                  {24 / 24,  60 * 4,   90 * 4},
              })
              // clang-format on
    );
}
