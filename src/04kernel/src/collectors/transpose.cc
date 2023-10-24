#include "kernel/collectors/transpose.h"
#include "../kernels/transpose/cpu_kernel.hh"

namespace refactor::kernel {

    TransposeCollector::TransposeCollector(
        Target target_,
        decltype(perm) perm_) noexcept
        : target(target_),
          perm(std::move(perm_)) {}

    std::vector<KernelBox>
    TransposeCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto info = TransposeInfo(inputs[0].get().shape, perm);

        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                if (auto ptr = TransposeCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case Target::NvidiaGpu:
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
