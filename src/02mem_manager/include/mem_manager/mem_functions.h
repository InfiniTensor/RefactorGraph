#ifndef MEM_MANAGER_MEM_FUNCTIONS_H
#define MEM_MANAGER_MEM_FUNCTIONS_H

#include <cstddef>

namespace refactor::mem_manager {

    using Malloc = void *(*) (size_t);
    using Free = void (*)(void *);
    using CopyHD = void (*)(void *, void *, size_t);// dst <- src ; n
    using CopyDH = void (*)(void *, void *, size_t);// dst <- src ; n
    using CopyDD = void (*)(void *, void *, size_t);// dst <- src ; n

    struct MemFunctions {
        Malloc malloc;
        Free free;
        CopyHD copyHd;
        CopyDH copyDh;
        CopyDD copyDd;
    };

}// namespace refactor::mem_manager

#endif// MEM_MANAGER_MEM_FUNCTIONS_H
