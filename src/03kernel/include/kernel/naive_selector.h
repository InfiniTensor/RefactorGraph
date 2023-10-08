#ifndef KERNEL_NAIVE_SELECTOR_H
#define KERNEL_NAIVE_SELECTOR_H

#include "selector.h"

namespace refactor::kernel {

    class NaiveSelector final : public Selector {
    public:
        std::unique_ptr<Kernel>
        select(std::vector<std::unique_ptr<Candidate>> const &,
               TensorRefs,
               TensorRefs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_NAIVE_SELECTOR_H
