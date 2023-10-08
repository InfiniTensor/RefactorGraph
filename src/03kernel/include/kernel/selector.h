#ifndef KERNEL_SELECTOR_H
#define KERNEL_SELECTOR_H

#include "candidata.h"

namespace refactor::kernel {

    class Selector {
    public:
        virtual std::unique_ptr<Kernel>
        select(std::vector<std::unique_ptr<Candidate>> const &,
               TensorRefs,
               TensorRefs) const = 0;
    };

}// namespace refactor::kernel

#endif// KERNEL_SELECTOR_H
