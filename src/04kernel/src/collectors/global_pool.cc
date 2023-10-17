#include "kernel/collectors/global_pool.h"
#include "refactor/common.h"

namespace refactor::kernel {

    GlobalPoolCollector::GlobalPoolCollector(
        Target target_,
        PoolType type_) noexcept
        : InfoCollector(),
          target(target_),
          type(type_) {}

    std::vector<KernelBox>
    GlobalPoolCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &x = inputs[0].get();
        auto const &y = outputs[0].get();

        auto rank = x.rank();
        absl::InlinedVector<uint16_t, 2> kernelShape(rank - 2);
        std::transform(x.shape.begin() + 2, x.shape.end(),
                       kernelShape.begin(),
                       [](auto dim) { return static_cast<uint16_t>(dim); });
        PoolAttributes attributes(rank - 2, nullptr, nullptr, nullptr);
        // TODO use pool kernels

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
