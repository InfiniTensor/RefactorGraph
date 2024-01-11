#include "kernel/collectors/where.h"
#include "../kernels/where/cnnl_kernel.hh"
#include "../kernels/where/cpu_kernel.hh"
#include "../kernels/where/where_cuda.hh"

namespace refactor::kernel {

    std::vector<KernelBox>
    WhereCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = WhereCpu::build(inputs); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = WhereCuda::build(inputs); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Mlu:
                if (auto ptr = WhereCnnl::build(inputs, outputs); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
