#include "kernel/collectors/moe.h"
#include "../kernels/moe/cpu_kernel.hh"
#include "kernel/attributes/moe_info.h"

namespace refactor::kernel {

    std::vector<KernelBox>
    AssignPosCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        AssignPosInfo info(topk, numExperts, inputs[0]);
        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = AssignPosCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            //todo ：暂时用cpu的实现
            case decltype(_target)::Nvidia:
                if (auto ptr = AssignPosCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

    std::vector<KernelBox>
    ReorderCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        ReorderInfo info(scatter, topk, inputs);
        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = ReorderCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            //todo ：暂时用cpu的实现
            case decltype(_target)::Nvidia:
                if (auto ptr = ReorderCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
