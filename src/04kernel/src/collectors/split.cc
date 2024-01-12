#include "kernel/collectors/split.h"
#include "../kernels/split/cnnl_kernel.hh"
#include "../kernels/split/cpu_kernel.hh"
#include "../kernels/split/cuda_kernel.hh"

namespace refactor::kernel {

    std::vector<KernelBox>
    SplitCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        SplitInfo info(axis, outputs);

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = SplitCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = SplitCuda::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Mlu:
                if (auto ptr = SplitCnnl::build(axis, inputs[0].get(), outputs); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
