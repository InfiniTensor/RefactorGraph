#include "memory.hh"
#include "functions.hh"

namespace refactor::hardware {
    using M = MluMemory;

    void *M::malloc(size_t size) {
        void *ptr;
        BANG_ASSERT(cnrtMalloc(&ptr, size));
        return ptr;
    }
    void M::free(void *ptr) {
        BANG_ASSERT(cnrtFree(ptr));
    }
    void *M::copyHD(void *dst, void const *src, size_t bytes) const {
        BANG_ASSERT(cnrtMemcpy(dst, const_cast<void *>(src), bytes,
                               CNRT_MEM_TRANS_DIR_HOST2DEV))
        return dst;
    }
    void *M::copyDH(void *dst, void const *src, size_t bytes) const {
        BANG_ASSERT(cnrtMemcpy(dst, const_cast<void *>(src), bytes,
                               CNRT_MEM_TRANS_DIR_DEV2HOST));
        return dst;
    }
    void *M::copyDD(void *dst, void const *src, size_t bytes) const {
        BANG_ASSERT(cnrtMemcpy(dst, const_cast<void *>(src), bytes,
                               CNRT_MEM_TRANS_DIR_PEER2PEER));
        return dst;
    }

}// namespace refactor::hardware
