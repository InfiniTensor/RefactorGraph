#include "../src/devices/cpu/memory.hh"
#include "hardware/mem_pool.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace hardware;

TEST(MemPool, testMallocFreeCpu) {
    auto memPool = MemPool(std::make_shared<CpuMemory>(), 4ul << 30, 8);
    auto a = 2 * 10 * 20 * 30 * sizeof(float),
         b = 3 * 10 * 20 * 30 * sizeof(float),
         c = 5 * 10 * 20 * 30 * sizeof(float),
         d = 7 * 10 * 20 * 30 * sizeof(float);
    // allocate a->b->c->d
    auto aPtr = memPool.malloc(a);
    auto bPtr = memPool.malloc(b);
    EXPECT_NE(aPtr, bPtr);
    auto cPtr = memPool.malloc(c);
    EXPECT_NE(bPtr, cPtr);
    auto dPtr = memPool.malloc(d);
    EXPECT_NE(cPtr, dPtr);
    // free b and c
    // expected to be a->mergedFreeBlock->d, where mergedFreeBlock is the result of
    // merging the memory blocks corresponding to the already freed b and c.
    memPool.free(bPtr);
    memPool.free(cPtr);
    // to obtain the starting address of a free block,
    // we simulate the allocation of an E tensor,
    // the size of which is the sum of the sizes of b and c.
    auto ePtr = memPool.malloc(b + c);
    EXPECT_EQ(ePtr, bPtr);
}
