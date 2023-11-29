#include "kernel/collectors/softmax.h"
#include "../kernels/softmax/cpu_kernel.hh"
#include "../kernels/softmax/cuda_kernel.hh"
#include "../kernels/softmax/cudnn_kernel.hh"
#include "kernel/attributes/softmax_info.h"

namespace refactor::kernel {

    std::vector<KernelBox>
    SoftmaxCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        SoftmaxInfo info(inputs[0].get(), axis);

        std::vector<KernelBox>
            ans;
        switch (_target) {
            case decltype(_target)::Cpu: {
                if (auto ptr = SoftmaxCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            }
            case decltype(_target)::Nvidia: {
                if (auto ptr = SoftmaxCudnn::build(cudnn::SoftmaxAlgo::ACCURATE, info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                if (auto ptr = SoftmaxCuda::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            }
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
