#include "kernel/collectors/concat.h"
#include "../kernels/concat/cpu_kernel.hh"
#include "../kernels/concat/cuda_kernel.hh"

namespace refactor::kernel {

    std::vector<KernelBox>
    ConcatCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        SplitInfo info(axis, inputs);

        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                if (auto ptr = ConcatCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case Target::NvidiaGpu:
                if (auto ptr = ConcatCuda::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
