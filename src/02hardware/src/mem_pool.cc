#include "hardware/mem_pool.h"

namespace refactor::hardware {

    MemPool::MemPool(decltype(_parent) parent, decltype(_memPoolSize) size, size_t alignment)
        : _parent(std::move(parent)),
          _memPoolSize(size),
          _calculator(OffsetCalculator(alignment)),
          _ptr(_parent->malloc(size)),
          _ptrToBlobsize{} {}
    MemPool::~MemPool() {
        _parent->free(_ptr);
    }
    void *MemPool::malloc(size_t const bytes) {
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
        return _parent->copyHD(dst, src, bytes);
    }
    void *MemPool::copyDH(void *dst, void const *src, size_t bytes) const noexcept {
        return _parent->copyDH(dst, src, bytes);
    }
    void *MemPool::copyDD(void *dst, void const *src, size_t bytes) const noexcept {
        return _parent->copyDD(dst, src, bytes);
    }

}// namespace refactor::hardware
