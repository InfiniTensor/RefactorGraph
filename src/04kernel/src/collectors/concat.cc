#include "kernel/collectors/concat.h"
#include "../kernels/concat/cpu_kernel.hh"
#include "../kernels/concat/cuda_kernel.hh"
#include "../kernels/concat/cnnl_kernel.hh"

namespace refactor::kernel {

    std::vector<KernelBox>
    ConcatCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        SplitInfo info(axis, inputs);

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = ConcatCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = ConcatCuda::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Mlu:
                if (auto ptr = ConcatCnnl::build(axis, inputs, outputs[0].get()); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
