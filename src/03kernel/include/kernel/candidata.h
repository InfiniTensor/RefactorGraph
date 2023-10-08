#ifndef KERNEL_CANDIDATE_H
#define KERNEL_CANDIDATE_H

#include "kernel.h"
#include "tensor.h"

namespace refactor::kernel {

    class Candidate {
    public:
        virtual bool compatible(
            TensorRefs const &inputs,
            TensorRefs const &outputs) const = 0;
        virtual std::unique_ptr<Kernel> lower() const = 0;
    };

    using CandidateBox = std::unique_ptr<Candidate>;

}// namespace refactor::kernel

#endif// KERNEL_CANDIDATE_H
