#include "kernel/collectors/pool.h"
#include "refactor/common.h"

namespace refactor::kernel {

    PoolCollector::PoolCollector(
        Target target_,
        PoolType type_,
        PoolAttributes attrs) noexcept
        : InfoCollector(),
          target(target_),
          type(type_),
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
