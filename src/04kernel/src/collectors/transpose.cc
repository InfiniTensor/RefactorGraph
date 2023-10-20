#include "kernel/collectors/transpose.h"

namespace refactor::kernel {

    TransposeCollector::TransposeCollector(
        Target target_,
        decltype(perm) perm_) noexcept
        : target(target_),
          perm(std::move(perm_)) {}

    std::vector<KernelBox>
    TransposeCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                break;
            case Target::NvidiaGpu:
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
