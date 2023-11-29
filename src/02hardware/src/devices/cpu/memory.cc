#include "memory.hh"
#include <cstdlib>
#include <cstring>

namespace refactor::hardware {
    using M = CpuMemory;

    void *M::malloc(size_t size) noexcept {
        return std::malloc(size);
    }
    void M::free(void *ptr) noexcept {
        std::free(ptr);
    }
    void *M::copyHD(void *dst, void const *src, size_t bytes) const noexcept {
        return std::memcpy(dst, src, bytes);
    }
    void *M::copyDH(void *dst, void const *src, size_t bytes) const noexcept {
        return std::memcpy(dst, src, bytes);
    }
    void *M::copyDD(void *dst, void const *src, size_t bytes) const noexcept {
        return std::memcpy(dst, src, bytes);
    }

}// namespace refactor::hardware
