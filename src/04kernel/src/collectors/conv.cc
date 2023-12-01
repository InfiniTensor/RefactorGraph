#include "kernel/collectors/conv.h"
#include "../kernels/conv/cudnn_kernel.hh"

namespace refactor::kernel {

    ConvCollector::ConvCollector(
        decltype(_target) target, PoolAttributes attrs) noexcept
        : InfoCollector(target), poolAttrs(std::move(attrs)) {}

    std::vector<KernelBox>
    ConvCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &x = inputs[0];
        auto const &w = inputs[1];
        auto b = inputs.size() == 3 ? std::make_optional(inputs[2]) : std::nullopt;
        auto const &y = outputs[0];

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = ConvCudnn::build(poolAttrs, x, w, b, y); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
