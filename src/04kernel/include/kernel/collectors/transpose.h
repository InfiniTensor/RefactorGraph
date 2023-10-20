#ifndef KERNEL_TRANSPOSE_H
#define KERNEL_TRANSPOSE_H

#include "../collector.h"
#include "../target.h"
#include "refactor/common.h"

namespace refactor::kernel {

    using Permutation = absl::InlinedVector<uint32_t, 4>;

    struct TransposeCollector final : public InfoCollector {
        Target target;
        Permutation perm;

        TransposeCollector(Target, decltype(perm)) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_TRANSPOSE_H
