#ifndef KERNEL_SOFTMAX_CNNL_HH
#define KERNEL_SOFTMAX_CNNL_HH

#include "kernel/attributes/softmax_info.h"
#include "kernel/collectors/softmax.h"

namespace refactor::kernel {

    namespace cnnl {
        enum class SoftmaxAlgo {
            FAST = 0,
            ACCURATE = 1,
            LOG = 2,
        };
    }// namespace cnnl

    struct SoftmaxCnnl final : public Kernel {
        cnnl::SoftmaxAlgo algo;
        DataType dataType;
        int pre, mid, post;

        SoftmaxCnnl(cnnl::SoftmaxAlgo, DataType, int, int, int) noexcept;

        static KernelBox build(cnnl::SoftmaxAlgo, SoftmaxInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const final;
#endif
    };

}// namespace refactor::kernel

#endif// KERNEL_SOFTMAX_CNNL_HH
