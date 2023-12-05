#include "kernel/collectors/all_reduce.h"
#include "../kernels/all_reduce/nccl_kernel.hh"
namespace refactor::kernel {
    std::vector<KernelBox>
    AllReduceCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = AllReduceNccl::build(type, inputs[0], outputs[0]); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }
}// namespace refactor::kernel