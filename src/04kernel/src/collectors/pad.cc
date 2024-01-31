#include "kernel/collectors/pad.h"
#include "../kernels/pad/cnnl_kernel.hh"
#include "../kernels/pad/cpu_kernel.hh"
#include "../kernels/pad/cuda_kernel.hh"

namespace refactor::kernel {

    std::vector<KernelBox>
    PadCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &input = inputs[0];
        PadInfo info(dims, input);
        auto const_value = inputs.size() >= 3 ? std::make_optional(inputs[2]) : std::nullopt;

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = PadCpu::build(std::move(info), mode, const_value); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = PadCuda::build(std::move(info), mode, const_value); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Mlu:
                if (auto ptr = PadCnnl::build(dims, input.get().dataType, mode, const_value); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
