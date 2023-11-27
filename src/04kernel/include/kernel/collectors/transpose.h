#ifndef KERNEL_TRANSPOSE_H
#define KERNEL_TRANSPOSE_H

#include "../attributes/transpose_info.h"
#include "../collector.h"

namespace refactor::kernel {

    struct TransposeCollector final : public InfoCollector {
        Permutation perm;

        TransposeCollector(decltype(_target), decltype(perm)) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_TRANSPOSE_H
