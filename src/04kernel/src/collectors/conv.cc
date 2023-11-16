#include "kernel/collectors/conv.h"
#include "../kernels/conv/cudnn_kernel.hh"

namespace refactor::kernel {

    ConvCollector::ConvCollector(
        Target target_,
        PoolAttributes attrs) noexcept
        : InfoCollector(),
          target(target_),
          poolAttrs(std::move(attrs)) {}

    std::vector<KernelBox>
    ConvCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &x = inputs[0].get();
        auto const &w = inputs[1].get();
        auto b = inputs.size() == 3 ? std::make_optional(inputs[2]) : std::nullopt;
        auto const &y = outputs[0].get();

        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                break;
            case Target::NvidiaGpu:
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
