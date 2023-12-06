#include "kernel/collectors/simple_unary.h"
#include "../kernels/simple_unary/cpu_kernel.hh"
#include "../kernels/simple_unary/cuda_kernel.hh"
#include "../kernels/simple_unary/cudnn_activation_kernel.hh"
#include "common.h"

namespace refactor::kernel {

#define REGISTER(T)                          \
    if (auto ptr = T::build(type, a); ptr) { \
        ans.emplace_back(std::move(ptr));    \
    }

    std::vector<KernelBox>
    SimpleUnaryCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &a = inputs[0];

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                REGISTER(SimpleUnaryCpu)
                break;
            case decltype(_target)::Nvidia:
                REGISTER(ActivationCudnn)
                REGISTER(SimpleUnaryCuda)
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
