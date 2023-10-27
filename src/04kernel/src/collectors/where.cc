#include "kernel/collectors/where.h"
#include "../kernels/where/cpu_kernel.hh"
#include "../kernels/where/where_cuda.hh"

namespace refactor::kernel {

#define REGISTER(T)                                  \
    if (auto ptr = T::build(c, x, y, output); ptr) { \
        ans.emplace_back(std::move(ptr));            \
    }

    std::vector<KernelBox>
    WhereCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &c = inputs[0].get();
        auto const &x = inputs[1].get();
        auto const &y = inputs[2].get();
        auto const &output = outputs[0].get();

        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                REGISTER(WhereCpu)
                break;
            case Target::NvidiaGpu:
                REGISTER(WhereCuda)
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
