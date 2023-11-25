#include "hardware/mem_pool.h"
#include "kernel/target.h"
#include "kernel/tensor.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(MemPool, testMallocFreeCpu) {
    auto a = Tensor::share(DataType::F32, Shape{10, 20, 20, 30});
    auto b = Tensor::share(DataType::F32, Shape{10, 40, 20, 30});
    auto c = Tensor::share(DataType::F32, Shape{10, 40, 20, 30});
    auto d = Tensor::share(DataType::F32, Shape{10, 20, 20, 30});
    Target target = Target::Cpu;
    auto memPool = target.memManager();
    // allocate a->b->c->d
    auto aPtr = memPool->malloc(a->bytesSize());
    auto bPtr = memPool->malloc(b->bytesSize());
    EXPECT_NE(aPtr, bPtr);
    auto cPtr = memPool->malloc(c->bytesSize());
    EXPECT_NE(bPtr, cPtr);
    auto dPtr = memPool->malloc(d->bytesSize());
    EXPECT_NE(cPtr, dPtr);
    // free b and c
    // expected to be a->mergedFreeBlock->d, where mergedFreeBlock is the result
    // of merging the memory blocks corresponding to the already freed b and c.
    memPool->free(bPtr);
    memPool->free(cPtr);
    // to obtain the starting address of a free block,
    // we simulate the allocation of an E tensor,
    // the size of which is the sum of the sizes of b and c.
    auto ePtr = memPool->malloc(b->bytesSize() + c->bytesSize());
    EXPECT_EQ(ePtr, bPtr);
}

#ifdef USE_CUDA
TEST(MemPool, testMallocFreeGpu) {
    auto a = Tensor::share(DataType::F32, Shape{10, 20, 20, 30});
    auto b = Tensor::share(DataType::F32, Shape{10, 40, 20, 30});
    auto c = Tensor::share(DataType::F32, Shape{10, 40, 20, 30});
    auto d = Tensor::share(DataType::F32, Shape{10, 20, 20, 30});
    Target target = Target::NvidiaGpu;
    auto memPool = target.memManager();
    // allocate a->b->c->d
    auto aPtr = memPool->malloc(a->bytesSize());
    auto bPtr = memPool->malloc(b->bytesSize());
    EXPECT_NE(aPtr, bPtr);
    auto cPtr = memPool->malloc(c->bytesSize());
    EXPECT_NE(bPtr, cPtr);
    auto dPtr = memPool->malloc(d->bytesSize());
    EXPECT_NE(cPtr, dPtr);
    // free b and c
    // expected to be a->mergedFreeBlock->d, where mergedFreeBlock is the result
    // of merging the memory blocks corresponding to the already freed b and c.
    memPool->free(bPtr);
    memPool->free(cPtr);
    // to obtain the starting address of a free block,
    // we simulate the allocation of an E tensor,
    // the size of which is the sum of the sizes of b and c.
    auto ePtr = memPool->malloc(b->bytesSize() + c->bytesSize());
    EXPECT_EQ(ePtr, bPtr);
}
#endif
