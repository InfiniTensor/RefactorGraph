#ifndef KERNEL_CANDIDATE_H
#define KERNEL_CANDIDATE_H

#include "kernel.h"
#include "tensor.h"

namespace refactor::kernel {

    class InfoCollector {
    public:
        virtual std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const = 0;
    };

    using CollectorBox = std::unique_ptr<InfoCollector>;

}// namespace refactor::kernel

#endif// KERNEL_CANDIDATE_H
