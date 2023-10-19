#include "kernel/collectors/simple_unary.h"
#include "../kernels/simple_unary/cpu_kernel.hh"
#include "../kernels/simple_unary/cudnn_activation_kernel.hh"
#include "refactor/common.h"

namespace refactor::kernel {

#define REGISTER(T)                          \
    if (auto ptr = T::build(type, a); ptr) { \
        ans.emplace_back(std::move(ptr));    \
    }

    std::vector<KernelBox>
    SimpleUnaryCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &a = inputs[0].get();

        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                REGISTER(SimpleUnaryCpu)
                break;
            case Target::NvidiaGpu:
                REGISTER(ActivationCudnn)
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
