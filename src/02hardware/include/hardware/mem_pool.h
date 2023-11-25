#ifndef MEM_MANAGER_MEM_POOL_H
#define MEM_MANAGER_MEM_POOL_H

#include "common.h"
#include "mem_manager.hh"
#include "mem_offset_calculator.h"
#include <unordered_map>

namespace refactor::hardware {

    class MemPool final : public hardware::MemManager {
        size_t _memPoolSize;
        OffsetCalculator _calculator;
        void *_ptr;
        std::unordered_map<void *, size_t> _ptrToBlobsize;
        Arc<MemManager> _f;

    public:
        void *malloc(size_t bytes) noexcept final;
        void free(void *ptr) final;
        void *copyHD(void *dst, void const *src, size_t bytes) const noexcept final;
        void *copyDH(void *dst, void const *src, size_t bytes) const noexcept final;
        void *copyDD(void *dst, void const *src, size_t bytes) const noexcept final;

        MemPool(size_t memPoolSize, size_t alignment, Arc<MemManager> f);
        ~MemPool();
    };

}// namespace refactor::hardware

#endif// MEM_MANAGER_MEM_POOL_H
