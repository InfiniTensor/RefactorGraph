#ifndef MEM_CALCULATOR_MEM_FUNCTIONS_H
#define MEM_CALCULATOR_MEM_FUNCTIONS_H

#include <cstddef>

namespace refactor::mem_manager {

    using CopyHD = void (*)(void *, void *, size_t);
    using CopyDH = void (*)(void *, void *, size_t);
    using CopyDD = void (*)(void *, void *, size_t);

    struct mem_functions {
        CopyHD copy_hd;
        CopyDH copy_dh;
        CopyDD copy_dd;
    };

}// namespace refactor::mem_manager

#endif// MEM_CALCULATOR_MEM_FUNCTIONS_H
