#include "kernel/collectors/dynamic_quantize_linear.h"
#include "../kernels/dynamic_quantize_linear/cpu_kernel.hh"
#include "../kernels/dynamic_quantize_linear/cuda_kernel.hh"

namespace refactor::kernel {

    DynamicQuantizeLinearCollector::
        DynamicQuantizeLinearCollector(decltype(_target) target) noexcept
        : InfoCollector(target) {}

    std::vector<KernelBox>
    DynamicQuantizeLinearCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto size = inputs[0].get().elementsSize();

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = DynamicQuantizeLinearCpu::build(size); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = DynamicQuantizeLinearCuda::build(size); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
