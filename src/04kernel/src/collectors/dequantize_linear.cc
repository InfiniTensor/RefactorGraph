#include "kernel/collectors/dequantize_linear.h"
#include "../kernels/dequantize_linear/cpu_kernel.hh"
#include "../kernels/dequantize_linear/cuda_kernel.hh"

namespace refactor::kernel {

    DequantizeLinearCollector::
        DequantizeLinearCollector(decltype(_target) target) noexcept
        : InfoCollector(target) {}

    std::vector<KernelBox>
    DequantizeLinearCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &output = outputs[0];
        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = DequantizeLinearCpu::build(inputs, output); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = DequantizeLinearCuda::build(inputs, output); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
