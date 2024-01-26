#include "kernel/collectors/hard_sigmoid.h"
#include "../kernels/hard_sigmoid/cpu_kernel.hh"
#include "../kernels/hard_sigmoid/cuda_kernel.hh"

namespace refactor::kernel {

    std::vector<KernelBox>
    HardSigmoidCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &a = inputs[0];

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = HardSigmoidCpu::build(alpha, beta, a); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = HardSigmoidCuda::build(alpha, beta, a); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
