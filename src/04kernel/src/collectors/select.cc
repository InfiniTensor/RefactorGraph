#include "kernel/collectors/select.h"
#include "../kernels/select/cpu_kernel.hh"
#include "../kernels/select/cuda_kernel.hh"
#include "../kernels/select/cnnl_kernel.hh"

namespace refactor::kernel {

#define REGISTER(T)                                     \
    if (auto ptr = T::build(selectType, inputs); ptr) { \
        ans.emplace_back(std::move(ptr));               \
    }

#define CASE(OP)         \
    case SelectType::OP: \
        return #OP

    std::string_view opName(SelectType type) {
        switch (type) {
            CASE(Max);
            CASE(Min);
            default:
                UNREACHABLE();
        }
    }

    SelectCollector::SelectCollector(decltype(_target) target, SelectType type) noexcept
        : InfoCollector(target), selectType(type) {}

    std::vector<KernelBox>
    SelectCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                REGISTER(SelectCpu)
                break;
            case decltype(_target)::Nvidia:
                REGISTER(SelectCuda)
                break;
            case decltype(_target)::Mlu:
                REGISTER(SelectCnnl)
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
