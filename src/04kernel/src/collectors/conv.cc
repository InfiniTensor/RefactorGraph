#include "kernel/collectors/conv.h"
#include "../kernels/conv/cudnn_kernel.hh"
#include "refactor/common.h"

namespace refactor::kernel {

    ConvCollector::ConvCollector(
        Target target_,
        PoolAttributes attrs) noexcept
        : InfoCollector(),
          target(target_),
          poolAttributes(std::move(attrs)) {}

#define REGISTER_CUDNN(ALGO)                   \
    if (auto ptr = ConvCudnn::build(           \
            (cudnn::ConvolutionFwdAlgo::ALGO), \
            poolAttributes,                    \
            w,                                 \
            x,                                 \
            y);                                \
        ptr) {                                 \
        ans.emplace_back(std::move(ptr));      \
    }

    std::vector<KernelBox>
    ConvCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &x = inputs[0].get();
        auto const &w = inputs[1].get();
        auto const &y = outputs[0].get();

        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                break;
            case Target::NvidiaGpu:
                REGISTER_CUDNN(IMPLICIT_GEMM)
                REGISTER_CUDNN(IMPLICIT_PRECOMP_GEMM)
                REGISTER_CUDNN(GEMM)
                REGISTER_CUDNN(DIRECT)
                REGISTER_CUDNN(FFT)
                REGISTER_CUDNN(FFT_TILING)
                REGISTER_CUDNN(WINOGRAD)
                REGISTER_CUDNN(WINOGRAD_NONFUSED)
                REGISTER_CUDNN(COUNT)
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
