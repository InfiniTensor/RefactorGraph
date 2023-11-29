#ifndef MEM_MANAGER_MEM_POOL_H
#define MEM_MANAGER_MEM_POOL_H

#include "common.h"
#include "mem_offset_calculator.h"
#include "memory.h"

namespace refactor::hardware {

    class MemPool final : public hardware::Memory {
        Arc<Memory> _parent;
        size_t _memPoolSize;

        OffsetCalculator _calculator;
        void *_ptr;
        std::unordered_map<void *, size_t> _ptrToBlobsize;

    public:
        MemPool(decltype(_parent), decltype(_memPoolSize), size_t alignment);
        ~MemPool();

        void *malloc(size_t bytes) final;
        void free(void *ptr) final;
        void *copyHD(void *dst, void const *src, size_t bytes) const noexcept final;
        void *copyDH(void *dst, void const *src, size_t bytes) const noexcept final;
        void *copyDD(void *dst, void const *src, size_t bytes) const noexcept final;
    };

}// namespace refactor::hardware

#endif// MEM_MANAGER_MEM_POOL_H
