#include "kernel/collectors/conv.h"
#include "../kernels/conv/cudnn_kernel.hh"
#include "common/error_handler.h"

namespace refactor::kernel {

    std::vector<KernelBox>
    ConvCollector::filter(TensorRefs inputs, TensorRefs outputs) const {

        std::vector<KernelBox> ans;
        if (auto ptr = ConvCudnn ::build(); ptr) {
            ans.emplace_back(std::move(ptr));
        }
        return ans;
    }

}// namespace refactor::kernel
