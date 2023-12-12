#include "kernel/collectors/cast.h"
#include "../kernels/cast/cpu_kernel.hh"
#include "../kernels/cast/cuda_kernel.hh"

namespace refactor::kernel {

    CastCollector::CastCollector(decltype(_target) target) noexcept
        : InfoCollector(target) {}

    std::vector<KernelBox>
    CastCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &from = inputs[0];
        auto const &to = outputs[0];

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = CastCpu::build(from, to); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = CastCuda::build(from, to); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
