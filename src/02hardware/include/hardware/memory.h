#ifndef HARDWARE_MEMORY_H
#define HARDWARE_MEMORY_H

#include <cstddef>

namespace refactor::hardware {

    class Memory {
    public:
        virtual ~Memory() = default;
        virtual void *malloc(size_t) = 0;
        virtual void free(void *) = 0;
        virtual void *copyHD(void *dst, void const *src, size_t bytes) const = 0;
        virtual void *copyDH(void *dst, void const *src, size_t bytes) const = 0;
        virtual void *copyDD(void *dst, void const *src, size_t bytes) const = 0;
    };

}// namespace refactor::hardware

#endif// HARDWARE_MEMORY_H
