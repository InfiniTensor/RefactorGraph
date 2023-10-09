#include "kernel/collectors/batch_normalization.h"
#include "../kernels/batch_normalization/batch_normalization_cudnn.hh"
#include "common/error_handler.h"

namespace refactor::kernel {

#define REGISTER(T)                                  \
    if (auto ptr = T::build(epsilon, inputs); ptr) { \
        ans.emplace_back(std::move(ptr));            \
    }

    std::vector<KernelBox>
    BatchNormalizationCollector::filter(TensorRefs inputs, TensorRefs outputs) const {

        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                break;
            case Target::NvidiaGpu:
                REGISTER(BatchNormalizationCudnn)
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
