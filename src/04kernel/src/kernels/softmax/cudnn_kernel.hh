#ifndef KERNEL_SOFTMAX_CUDNN_HH
#define KERNEL_SOFTMAX_CUDNN_HH

#include "kernel/attributes/softmax_info.h"
#include "kernel/collectors/softmax.h"

namespace refactor::kernel {

    namespace cudnn {
        enum class SoftmaxAlgo {
            FAST = 0,
            ACCURATE = 1,
            LOG = 2,
        };
    }// namespace cudnn

    struct SoftmaxCudnn final : public Kernel {
        cudnn::SoftmaxAlgo algo;
        DataType dataType;
        std::vector<int> dim;

        SoftmaxCudnn(cudnn::SoftmaxAlgo, DataType, std::vector<int>) noexcept;

        static KernelBox build(cudnn::SoftmaxAlgo, SoftmaxInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_CUDA
        Routine lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_SOFTMAX_CUDNN_HH
