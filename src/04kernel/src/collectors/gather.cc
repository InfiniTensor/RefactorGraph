#include "kernel/collectors/gather.h"
#include "../kernels/gather/cpu_kernel.hh"
#include "../kernels/gather/cuda_kernel.hh"
namespace refactor::kernel {
#define REGISTER(T)                                              \
    if (auto ptr = T::build(data, indices, output, axis); ptr) { \
        ans.emplace_back(std::move(ptr));                        \
    }

    std::vector<KernelBox>
    GatherCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        std::vector<KernelBox> ans;
        auto const &data = inputs[0].get();
        auto const &indices = inputs[1].get();
        auto const &output = outputs[0].get();
        switch (target) {
            case Target::Cpu:
                REGISTER(GatherCpu)
                break;
            case Target::NvidiaGpu:
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
