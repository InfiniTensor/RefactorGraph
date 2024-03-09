#include "kernel/collectors/topk.h"
#include "../kernels/topk/cpu_kernel.hh"
#include "kernel/attributes/topk_info.h"
//#include "../kernels/topk/cuda_kernel.hh"

namespace refactor::kernel {

    std::vector<KernelBox>
    TopKCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        TopKInfo info(topk, axis, inputs[0]);
        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = TopKCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            //todo ：暂时用cpu的实现
            case decltype(_target)::Nvidia:
                if (auto ptr = TopKCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
