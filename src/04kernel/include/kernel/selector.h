#ifndef KERNEL_SELECTOR_H
#define KERNEL_SELECTOR_H

#include "kernel.h"

namespace refactor::kernel {

    class Selector {
    public:
        virtual ~Selector() = default;
        virtual KernelBox select(std::vector<KernelBox>) const = 0;
    };

}// namespace refactor::kernel

#endif// KERNEL_SELECTOR_H
