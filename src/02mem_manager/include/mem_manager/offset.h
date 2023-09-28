#ifndef MEM_MANAGER_OFFSET_H
#define MEM_MANAGER_OFFSET_H

#include "segmentation.h"
#include <memory>

namespace refactor::mem_manager {

    struct Offset {
        std::shared_ptr<Segmentation> segmentation;
        size_t offset;
    };

}// namespace refactor::mem_manager

#endif// MEM_MANAGER_OFFSET_H
