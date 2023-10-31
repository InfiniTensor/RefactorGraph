#include "kernel/attributes/broadcaster.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(kernel, Broadcaster) {
    // _ 3 5 7 -> 1 3 35 ->   0 35 1
    // _ _ 5 7 -> 1 1 35 ->   0  0 1
    // _ _ _ _ -> 1 1  1 ->   0  0 0
    // 2 1 1 1 -> 2 1  1 ->   1  0 0
    // 2 3 5 7 -> 2 3 35 -> 105 35 1
    Shape
        s0{3, 5, 7},
        s1{5, 7},
        s2{},
        s3{2, 1, 1, 1};
    Broadcaster broadcaster({
        slice(s0.data(), s0.size()),
        slice(s1.data(), s1.size()),
        slice(s2.data(), s2.size()),
        slice(s3.data(), s3.size()),
    });
    EXPECT_EQ(broadcaster.outputsCount, 210);
    EXPECT_EQ(broadcaster.strides,
              (std::vector<uint_lv2>{
                  0, 0, 0, 1, 105,
                  35, 0, 0, 0, 35,
                  1, 1, 0, 0, 1}));

    std::vector<uint_lv2> ans(broadcaster.inputsCount);
    broadcaster.locate(1, ans.data());
    EXPECT_EQ(ans, (std::vector<uint_lv2>{1, 1, 0, 0}));
    broadcaster.locate(40, ans.data());
    EXPECT_EQ(ans, (std::vector<uint_lv2>{40, 5, 0, 0}));
}
