#include "kernel/collectors/transpose.h"
#include "../kernels/transpose/cpu_kernel.hh"
#include "../kernels/transpose/cuda_kernel.hh"

namespace refactor::kernel {

    TransposeCollector::TransposeCollector(
        decltype(_target) target, decltype(perm) perm_) noexcept
        : InfoCollector(target), perm(std::move(perm_)) {}

    std::vector<KernelBox>
    TransposeCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &data = inputs[0].get();
        auto info = TransposeInfo(data.shape, perm);

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = TransposeCpu::build(data.dataType, info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = TransposeCuda::build(data.dataType, info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
