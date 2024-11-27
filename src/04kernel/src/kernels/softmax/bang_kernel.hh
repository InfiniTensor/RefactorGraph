#ifndef KERNEL_SOFTMAX_BANG_HH
#define KERNEL_SOFTMAX_BANG_HH

#include "cnnl.h"
#include "cnrt.h"
#include "kernel/attributes/softmax_info.h"
#include "kernel/collectors/softmax.h"
namespace refactor::kernel {

    struct SoftmaxBang final : public Kernel {
        SoftmaxInfo info;

        SoftmaxBang(SoftmaxInfo) noexcept;
        static KernelBox build(SoftmaxInfo) noexcept;
        static size_t typeId() noexcept;

        size_t kernelTypeId() const noexcept final;
        std::string_view description() const noexcept final;
#ifdef USE_BANG
        RoutineWorkspace lower(Resources &) const noexcept final;
#endif
    };

}// namespace refactor::kernel

#endif//KERNEL_SOFTMAX_BANG_HH
