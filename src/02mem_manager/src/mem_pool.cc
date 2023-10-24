#include "mem_manager/mem_pool.h"

namespace refactor::mem_manager {
    MemPool::MemPool(size_t memPoolSize, size_t alignment, Arc<MemManager> f)
        : _memPoolSize(memPoolSize), _calculator(OffsetCalculator(alignment)), _ptr(f->malloc(memPoolSize)), _f(f) {
    }
    MemPool::~MemPool() {
        _f->free(_ptr);
    }
    void *MemPool::malloc(size_t bytes) noexcept {
        size_t offset = _calculator.alloc(bytes);
        ASSERT(_calculator.getPeek() < _memPoolSize, "out of memory");
        void *retPtr = static_cast<uint8_t *>(_ptr) + offset;
        _ptrToBlobsize[retPtr] = bytes;
        return retPtr;
    }
    void MemPool::free(void *ptr) noexcept {
        ASSERT(_ptrToBlobsize.find(ptr) != _ptrToBlobsize.end(), "invalid ptr");
        size_t size = _ptrToBlobsize[ptr];
        size_t offset = static_cast<uint8_t *>(ptr) - static_cast<uint8_t *>(_ptr);
        _calculator.free(offset, size);
        _ptrToBlobsize.erase(ptr);
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