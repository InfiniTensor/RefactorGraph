#include "../../../02mem_manager/include/mem_manager/mem_offset_calculator.h"
#include "../../include/kernel/tensor.h"
#include <gtest/gtest.h>

using namespace refactor;
using namespace kernel;

TEST(MemOffsetCalculator, testMergeFreeBlocks) {
    Shape shape = Shape{10, 20, 20, 30};
    auto a = Tensor::share(DataType::F32, shape);
    auto b = Tensor::share(DataType::F32, shape);
    auto c = Tensor::share(DataType::F32, shape);
    auto d = Tensor::share(DataType::F32, shape);
    mem_manager::OffsetCalculator calculator = mem_manager::OffsetCalculator(sizeof(uint64_t));
    // allocate a->b->c->d
    calculator.alloc(a->bytesSize());
    size_t offsetB = calculator.alloc(b->bytesSize());
    size_t offsetC = calculator.alloc(c->bytesSize());
    calculator.alloc(d->bytesSize());
    // free b and c
    // expected to be a->mergedFreeBlock->d, where mergedFreeBlock is the result
    // of merging the memory blocks corresponding to the already freed b and c.
    calculator.free(offsetB, b->bytesSize());
    calculator.free(offsetC, c->bytesSize());
    // to obtain the starting address of a free block,
    // we simulate the allocation of an E tensor,
    // the size of which is the sum of the sizes of b and c.
    size_t offsetE = calculator.alloc(b->bytesSize() + c->bytesSize());
    EXPECT_EQ(offsetE, offsetB);
}

TEST(MemOffsetCalculator, testAllocFree) {
    Shape shape = Shape{10, 20, 20, 30};
    auto a = Tensor::share(DataType::F32, shape);
    auto b = Tensor::share(DataType::F32, shape);
    auto c = Tensor::share(DataType::F32, shape);
    auto d = Tensor::share(DataType::F32, shape);
    mem_manager::OffsetCalculator calculator = mem_manager::OffsetCalculator(sizeof(uint64_t));
    // allocate a->b->c
    calculator.alloc(a->bytesSize());
    size_t offsetB = calculator.alloc(b->bytesSize());
    calculator.alloc(c->bytesSize());
    // free b, then allocate d
    calculator.free(offsetB, b->bytesSize());
    size_t offsetC = calculator.alloc(d->bytesSize());
    // expected to be a->d->c
    EXPECT_EQ(offsetB, offsetC);
}

TEST(MemOffsetCalculator, testAllocWithEndFreeBlock) {
    Shape shape = Shape{10, 20, 20, 30};
    auto a = Tensor::share(DataType::F32, shape);
    auto b = Tensor::share(DataType::F32, shape);
    auto c = Tensor::share(DataType::F32, shape);
    auto d =
        Tensor::share(DataType::F32, Shape{20, 20, 20, 30});
    mem_manager::OffsetCalculator calculator = mem_manager::OffsetCalculator(sizeof(uint64_t));
    // allocate a->b->c
    calculator.alloc(a->bytesSize());
    calculator.alloc(b->bytesSize());
    size_t offsetC = calculator.alloc(c->bytesSize());
    // free c, then allocate d
    calculator.free(offsetC, c->bytesSize());
    size_t offsetD = calculator.alloc(d->bytesSize());
    // expected to be a->b->d, with no free block between b and d
    EXPECT_EQ(offsetC, offsetD);
}
