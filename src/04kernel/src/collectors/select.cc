#include "kernel/collectors/select.h"

namespace refactor::kernel {

    SelectCollector::SelectCollector(decltype(_target) target, SelectType type) noexcept
        : InfoCollector(target), selectType(type) {}

    std::vector<KernelBox>
    SelectCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
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
