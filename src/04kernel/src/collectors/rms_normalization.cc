#include "kernel/collectors/rms_normalization.h"
#include "../kernels/rms_normalization/cpu_kernel.hh"
#include "../kernels/rms_normalization/cuda_kernel.hh"

namespace refactor::kernel {

#define REGISTER(T)                                  \
    if (auto ptr = T::build(epsilon, inputs); ptr) { \
        ans.emplace_back(std::move(ptr));            \
    }

    std::vector<KernelBox>
    RmsNormalizationCollector::filter(TensorRefs inputs, TensorRefs outputs) const {

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                REGISTER(RmsNormalizationCpu)
                break;
            case decltype(_target)::Nvidia:
                REGISTER(RmsNormalizationCuda)
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
