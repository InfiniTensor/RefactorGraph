#include "kernel/collectors/attention.h"
#include "kernel/kernel.h"
#include "kernel/tensor.h"
// #include "../kernels/attention/cpu_kernel.hh"
// #include "../kernels/attention/cuda_kernel.hh"

namespace refactor::kernel {

    AttentionCollector::AttentionCollector(
        decltype(_target) target,
        decltype(maxSeqLen) maxSeqLen_) noexcept
        : InfoCollector(target),
          maxSeqLen(maxSeqLen_) {}

    std::vector<KernelBox>
    AttentionCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                break;
            case decltype(_target)::Nvidia:
                break;
            case decltype(_target)::Mlu:
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
