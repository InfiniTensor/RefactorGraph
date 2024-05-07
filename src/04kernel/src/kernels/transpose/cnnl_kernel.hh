#ifndef KERNEL_TRANSPOSE_CNNL_KERNEL_HH
#define KERNEL_TRANSPOSE_CNNL_KERNEL_HH

#include "kernel/collectors/transpose.h"
#include "kernel/tensor.h"

namespace refactor::kernel {

    using Shape = absl::InlinedVector<dim_t, 4>;
    using Permutation = Shape;

    struct TransposeInfoCnnl {
        DataType dataType;
        std::vector<int> inDim, outDim, perm;

        TransposeInfoCnnl(DataType, std::vector<int>, std::vector<int>);
        TransposeInfoCnnl(DataType, Shape, Permutation);
    };

    struct TransposeCnnl final : public Kernel {
        TransposeInfoCnnl info;

        TransposeCnnl(TransposeInfoCnnl) noexcept;

        static KernelBox build(DataType, Shape, Permutation) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_TRANSPOSE_CNNL_KERNEL_HH
