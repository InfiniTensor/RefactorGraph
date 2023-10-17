#include "kernel/collectors/pool.h"
#include "refactor/common.h"

namespace refactor::kernel {

    PoolCollector::PoolCollector(
        Target target_,
        PoolType type_,
        bool ceil_,
        decltype(kernelShape) kernelShape_,
        PoolAttributes attrs) noexcept
        : InfoCollector(),
          target(target_),
          type(type_),
          ceil(ceil_),
          kernelShape(std::move(kernelShape_)),
          attributes(std::move(attrs)) {}

    std::vector<KernelBox>
    PoolCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
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
