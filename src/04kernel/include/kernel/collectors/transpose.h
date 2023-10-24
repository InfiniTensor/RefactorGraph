#ifndef KERNEL_TRANSPOSE_H
#define KERNEL_TRANSPOSE_H

#include "../attributes/transpose_info.h"
#include "../collector.h"
#include "../target.h"

namespace refactor::kernel {

    struct TransposeCollector final : public InfoCollector {
        Target target;
        Permutation perm;

        TransposeCollector(Target, decltype(perm)) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_TRANSPOSE_H
