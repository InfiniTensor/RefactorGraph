#ifndef KERNEL_CONV_CUDNN_KERNEL_HH
#define KERNEL_CONV_CUDNN_KERNEL_HH

#include "kernel/attributes/expand_info.h"
#include "kernel/attributes/pool_attributes.h"
#include "kernel/kernel.h"
#include <optional>

namespace refactor::kernel {

    /// @brief Use `cudnnConvolutionForward`.
    ///        It only supports 4D tensors.
    struct ConvCudnn final : public Kernel {
        struct {
            DataType dt;
            int xShape[4],
                wShape[4],
                yShape[4],
                dilation[2],
                pad[2],
                stride[2];
        } info;
        std::optional<ExpandInfo> biasExpand;

        explicit ConvCudnn(decltype(info), decltype(biasExpand)) noexcept;

        static KernelBox build(PoolAttributes const &,
                               Tensor const &,
                               Tensor const &,
                               std::optional<std::reference_wrapper<Tensor const>>,
                               Tensor const &);
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_CONV_CUDNN_KERNEL_HH
