#ifndef MEM_CALCULATOR_SEGMENTATION_H
#define MEM_CALCULATOR_SEGMENTATION_H

#include "mem_functions.h"

namespace refactor::mem_manager {

    struct Segmentation {
        mem_functions functions;

        void *ptr;
        size_t size;
    };

}// namespace refactor::mem_manager

#endif// MEM_CALCULATOR_SEGMENTATION_H
