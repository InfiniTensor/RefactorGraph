#include "hardware/mem_offset_calculator.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace hardware;

TEST(MemOffsetCalculator, testMergeFreeBlocks) {
    auto a = 10 * 20 * 30 * 40 * sizeof(float),
         b = 10 * 20 * 30 * 40 * sizeof(float),
         c = 10 * 20 * 30 * 40 * sizeof(float),
         d = 10 * 20 * 30 * 40 * sizeof(float);
    OffsetCalculator calculator(sizeof(uint64_t));
    // allocate a->b->c->d
    calculator.alloc(a);
    size_t offsetB = calculator.alloc(b);
    size_t offsetC = calculator.alloc(c);
    calculator.alloc(d);
    // free b and c
    // expected to be a->mergedFreeBlock->d, where mergedFreeBlock is the result
    // of merging the memory blocks corresponding to the already freed b and c.
    calculator.free(offsetB, b);
    calculator.free(offsetC, c);
    // to obtain the starting address of a free block,
    // we simulate the allocation of an E tensor,
    // the size of which is the sum of the sizes of b and c.
    size_t offsetE = calculator.alloc(b + c);
    EXPECT_EQ(offsetE, offsetB);
}

TEST(MemOffsetCalculator, testAllocFree) {
    auto a = 10 * 20 * 30 * 40 * sizeof(float),
         b = 10 * 20 * 30 * 40 * sizeof(float),
         c = 10 * 20 * 30 * 40 * sizeof(float),
         d = 10 * 20 * 30 * 40 * sizeof(float);
    OffsetCalculator calculator(sizeof(uint64_t));
    // allocate a->b->c
    calculator.alloc(a);
    size_t offsetB = calculator.alloc(b);
    calculator.alloc(c);
    // free b, then allocate d
    calculator.free(offsetB, b);
    size_t offsetC = calculator.alloc(d);
    // expected to be a->d->c
    EXPECT_EQ(offsetB, offsetC);
}

TEST(MemOffsetCalculator, testAllocWithEndFreeBlock) {
    auto a = 10 * 20 * 30 * 40 * sizeof(float),
         b = 10 * 20 * 30 * 40 * sizeof(float),
         c = 10 * 20 * 30 * 40 * sizeof(float),
         d = 20 * 20 * 30 * 40 * sizeof(float);
    OffsetCalculator calculator(sizeof(uint64_t));
    // allocate a->b->c
    calculator.alloc(a);
    calculator.alloc(b);
    size_t offsetC = calculator.alloc(c);
    // free c, then allocate d
    calculator.free(offsetC, c);
    size_t offsetD = calculator.alloc(d);
    // expected to be a->b->d, with no free block between b and d
    EXPECT_EQ(offsetC, offsetD);
}
