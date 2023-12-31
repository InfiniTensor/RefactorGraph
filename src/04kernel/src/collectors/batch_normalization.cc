﻿#include "kernel/collectors/batch_normalization.h"
#include "../kernels/batch_normalization/cpu_kernel.hh"
#include "../kernels/batch_normalization/cudnn_kernel.hh"

namespace refactor::kernel {

#define REGISTER(T)                                  \
    if (auto ptr = T::build(epsilon, inputs); ptr) { \
        ans.emplace_back(std::move(ptr));            \
    }

    std::vector<KernelBox>
    BatchNormalizationCollector::filter(TensorRefs inputs, TensorRefs outputs) const {

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                REGISTER(BatchNormalization)
                break;
            case decltype(_target)::Nvidia:
                REGISTER(BatchNormalizationCudnn)
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
