﻿#include "kernel/collectors/global_pool.h"
#include "../kernels/pool/cudnn_kernel.hh"

namespace refactor::kernel {

    GlobalPoolCollector::GlobalPoolCollector(
        decltype(_target) target, PoolType type_) noexcept
        : InfoCollector(target), type(type_) {}

    std::vector<KernelBox>
    GlobalPoolCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &x = inputs[0].get();
        auto const &y = outputs[0];

        auto rank = x.rank() - 2;
        KernelShape kernelShape(rank);
        std::transform(x.shape.begin() + 2, x.shape.end(),
                       kernelShape.begin(),
                       [](auto dim) { return static_cast<ddim_t>(dim); });
        PoolAttributes attributes(rank, nullptr, nullptr, nullptr);

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = PoolCudnn::build(type, false, kernelShape, attributes, x, y); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
