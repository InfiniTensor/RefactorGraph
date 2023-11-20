#include "mem_manager/mem_pool.h"

namespace refactor::mem_manager {
    MemPool::MemPool(size_t memPoolSize, size_t alignment, Arc<MemManager> f)
        : _memPoolSize(memPoolSize),
          _calculator(OffsetCalculator(alignment)),
          _ptr(f->malloc(memPoolSize)),
          _f(std::move(f)) {
    }
    MemPool::~MemPool() {
        _f->free(_ptr);
    }
    void *MemPool::malloc(size_t const bytes) noexcept {
        if (bytes == 0) { return nullptr; }
        auto offset = _calculator.alloc(bytes);
        ASSERT(_calculator.peak() < _memPoolSize, "out of memory");
        void *ans = static_cast<uint8_t *>(_ptr) + offset;
        _ptrToBlobsize.emplace(ans, bytes);
        return ans;
    }
    void MemPool::free(void *const ptr) {
        if (!ptr) { return; }
        auto it = _ptrToBlobsize.find(ptr);
        ASSERT(it != _ptrToBlobsize.end(), "invalid ptr");
        auto offset = static_cast<uint8_t *>(ptr) - static_cast<uint8_t *>(_ptr);
        _calculator.free(offset, it->second);
        _ptrToBlobsize.erase(it);
    }
    void *MemPool::copyHD(void *dst, void const *src, size_t bytes) const noexcept {
        _f->copyHD(dst, src, bytes);
        return dst;
    }
    void *MemPool::copyDH(void *dst, void const *src, size_t bytes) const noexcept {
        _f->copyDH(dst, src, bytes);
        return dst;
    }
    void *MemPool::copyDD(void *dst, void const *src, size_t bytes) const noexcept {
        _f->copyDD(dst, src, bytes);
        return dst;
    }
}// namespace refactor::mem_manager
