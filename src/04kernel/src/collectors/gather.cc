#include "kernel/collectors/gather.h"
#include "../kernels/gather/cnnl_kernel.hh"
#include "../kernels/gather/cpu_kernel.hh"
#include "../kernels/gather/cuda_kernel.hh"

namespace refactor::kernel {

    std::vector<KernelBox>
    GatherCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        GatherInfo info(axis, inputs[0], inputs[1]);

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = GatherCpu::build(info); ptr != nullptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = GatherCuda::build(info); ptr != nullptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Mlu:
                if (auto ptr = GatherCnnl::build(axis, inputs[0].get(), inputs[1].get(), outputs[0].get()); ptr != nullptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
