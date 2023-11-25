#ifndef MEM_MANAGER_MEM_FUNCTIONS_H
#define MEM_MANAGER_MEM_FUNCTIONS_H

#include <cstddef>

namespace refactor::hardware {

    class MemManager {
    public:
        virtual ~MemManager() = default;
        virtual void *malloc(size_t) = 0;
        virtual void free(void *) = 0;
        virtual void *copyHD(void *dst, void const *src, size_t bytes) const = 0;
        virtual void *copyDH(void *dst, void const *src, size_t bytes) const = 0;
        virtual void *copyDD(void *dst, void const *src, size_t bytes) const = 0;
    };

}// namespace refactor::hardware

#endif// MEM_MANAGER_MEM_FUNCTIONS_H
