#include "kernel/collectors/mat_mul.h"
#include "refactor/common.h"

namespace refactor::kernel {

    std::vector<KernelBox>
    MatMulCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &a = inputs[0].get();
        auto const &b = inputs[1].get();
        auto const &c = outputs[0].get();

        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                break;
            case Target::NvidiaGpu:
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
