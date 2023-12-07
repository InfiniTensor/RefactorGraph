#include "kernel/collectors/cast.h"

namespace refactor::kernel {

    CastCollector::CastCollector(decltype(_target) target) noexcept
        : InfoCollector(target) {}

    std::vector<KernelBox>
    CastCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                break;
            case decltype(_target)::Nvidia:
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
