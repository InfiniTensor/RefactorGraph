#include "kernel/collectors/clip.h"
#include "../kernels/clip/cpu_kernel.hh"
#include "../kernels/clip/cuda_kernel.hh"

namespace refactor::kernel {

    ClipCollector::ClipCollector(decltype(_target) target) noexcept
        : InfoCollector(target) {}

    std::vector<KernelBox>
    ClipCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &data = inputs[0];
        auto hasMax = inputs.size() == 3;

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = ClipCpu::build(data, hasMax); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = ClipCuda::build(data, hasMax); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
