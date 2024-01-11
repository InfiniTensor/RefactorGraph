#ifndef KERNEL_CONV_CNNL_KERNEL_HH
#define KERNEL_CONV_CNNL_KERNEL_HH

#include "../../kernels/expand/cnnl_kernel.hh"
#include "kernel/attributes/pool_attributes.h"
#include "kernel/kernel.h"
#include <optional>

namespace refactor::kernel {

    /// @brief Use `cnnlConvolutionForward`.
    ///        It only supports 4D tensors.
    struct ConvCnnl final : public Kernel {
        struct {
            DataType dt;
            int xShape[4],
                wShape[4],
                yShape[4],
                dilation[2],
                pad[4],
                stride[2];
            std::optional<ExpandInfoCnnl> biasExpand;
        } info;

        explicit ConvCnnl(decltype(info)) noexcept;

        static KernelBox build(PoolAttributes const &,
                               Tensor const &,
                               Tensor const &,
                               std::optional<std::reference_wrapper<Tensor const>>,
                               Tensor const &);
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_CONV_CNNL_KERNEL_HH
