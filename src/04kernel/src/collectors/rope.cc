#include "kernel/collectors/rope.h"
#include "kernel/attributes/rope_info.h"
#include "../kernels/rope/cpu_kernel.hh"
#include "../kernels/rope/cuda_kernel.hh"
namespace refactor::kernel {
    std::vector<KernelBox>
    RoPECollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        RoPEInfo info(inputs[0], theta);

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = RoPECpu::build(info, inputs[0]); ptr != nullptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = RoPECuda::build(info, inputs[0]); ptr != nullptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }
}// namespace refactor::kernel
