#include "kernel/collectors/reduce.h"
#include "../kernels/reduce/cpu_kernel.hh"
#include "../kernels/reduce/cudnn_kernel.hh"

namespace refactor::kernel {

#define REGISTER(T)                                           \
    if (auto ptr = T::build(axes, reduceType, inputs); ptr) { \
        ans.emplace_back(std::move(ptr));                     \
    }

    ReduceCollector::ReduceCollector(
        Target targets_,
        ReduceType type_,
        Axes axes_) noexcept
        : InfoCollector(),
          target(targets_),
          reduceType(type_),
          axes(std::move(axes_)) {}

    std::vector<KernelBox>
    ReduceCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                REGISTER(ReduceCpu)
                break;
            case Target::NvidiaGpu:
                REGISTER(ReduceCudnn)
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel