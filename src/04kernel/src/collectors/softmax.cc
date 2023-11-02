#include "kernel/collectors/softmax.h"
#include "../kernels/softmax/cpu_kernel.hh"
#include "../kernels/softmax/cudnn_kernel.hh"

namespace refactor::kernel {

#define REGISTER_CUDNN(ALGO, MODE)        \
    if (auto ptr = SoftmaxCudnn::build(   \
            (cudnn::SoftmaxAlgo::ALGO),   \
            (cudnn::SoftmaxMode::MODE),   \
            i);                           \
        ptr) {                            \
        ans.emplace_back(std::move(ptr)); \
    }

    std::vector<KernelBox>
    SoftmaxCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        //auto const &o = outputs[0].get();

        std::vector<KernelBox>
            ans;
        switch (target) {
            case Target::Cpu: {
                AxisInfo info(inputs[0].get(), axis);
                if (auto ptr = SoftmaxCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            }
            case Target::NvidiaGpu: {
                auto const &i = inputs[0].get();
                if (axis == 1) {
                    REGISTER_CUDNN(FAST, CHANNEL)
                    REGISTER_CUDNN(ACCURATE, CHANNEL)
                }
                break;
            }
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel