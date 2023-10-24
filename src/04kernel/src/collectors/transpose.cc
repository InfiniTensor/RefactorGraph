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
        auto const &data = inputs[0].get();
        auto info = TransposeInfo(data.shape, perm);

        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                if (auto ptr = TransposeCpu::build(data.dataType, info); ptr) {
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
