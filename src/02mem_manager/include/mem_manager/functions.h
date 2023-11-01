#ifndef MEM_MANAGER_FUNCTIONS_H
#define MEM_MANAGER_FUNCTIONS_H

#include <cstddef>

namespace refactor::mem_manager {

    constexpr size_t align2power(size_t size, int bits) {
        auto mask = (1 << bits) - 1;
        return (size + mask) & ~mask;
    }

    constexpr size_t alignBytes(size_t size, int bytes) {
        return (size + bytes - 1) / bytes * bytes;
    }

}// namespace refactor::mem_manager

#endif// MEM_MANAGER_FUNCTIONS_H
