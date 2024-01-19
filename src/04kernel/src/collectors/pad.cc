#include "../kernels/pad/cpu_kernel.hh"
// #include "../kernels/pad/cuda_kernel.hh"
#include "kernel/collectors/pad.h"

namespace refactor::kernel {

    std::vector<KernelBox>
    PadCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &input = inputs[0];
        auto const &output = outputs[0];
        bool have_value = inputs.size() >= 3 ? true : false;
        PadInfo info(pads, mode, input, output, have_value);

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = PadCpu::build(std::move(info)); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            // case decltype(_target)::Nvidia:
            //     if (auto ptr = PadCuda::build(); ptr) {
            //         ans.emplace_back(std::move(ptr));
            //     }
            //     break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel