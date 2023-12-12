#include "kernel/collectors/scatter_nd.h"
#include "../kernels/scatter_nd/cpu_kernel.hh"
#include "../kernels/scatter_nd/cuda_kernel.hh"

namespace refactor::kernel {

    ScatterNDCollector::ScatterNDCollector(decltype(_target) target) noexcept
        : InfoCollector(target) {}

    std::vector<KernelBox>
    ScatterNDCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        ScatterNDInfo info(inputs[0], inputs[1]);

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = ScatterNDCpu::build(std::move(info)); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = ScatterNDCuda::build(std::move(info)); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
