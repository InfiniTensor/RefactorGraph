#include "kernel/collectors/softmax.h"
#include "../kernels/softmax/cpu_kernel.hh"

namespace refactor::kernel {

#define REGISTER(T)                          \
    if (auto ptr = T::build(info, i); ptr) { \
        ans.emplace_back(std::move(ptr));    \
    }

    std::vector<KernelBox>
    SoftmaxCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        AxisInfo info(inputs[0].get(), axis);
        auto const &i = inputs[0].get();
        //auto const &o = outputs[0].get();

        std::vector<KernelBox>
            ans;
        switch (target) {
            case Target::Cpu:
                //REGISTER(SoftmaxCpu)
                break;
            case Target::NvidiaGpu:
                //REGISTER(SoftmaxCuda)
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel