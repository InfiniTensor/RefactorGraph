#include "kernel/collectors/pool.h"
#include "../kernels/pool/cudnn_kernel.hh"

namespace refactor::kernel {

    PoolCollector::PoolCollector(
        Target target_,
        PoolType type_,
        bool ceil_,
        KernelShape kernelShape_,
        PoolAttributes attrs) noexcept
        : InfoCollector(),
          target(target_),
          type(type_),
          ceil(ceil_),
          kernelShape(std::move(kernelShape_)),
          attributes(std::move(attrs)) {}

    std::vector<KernelBox>
    PoolCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &x = inputs[0].get();
        auto const &y = outputs[0].get();

        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                break;
            case Target::NvidiaGpu:
                if (auto ptr = PoolCudnn::build(type, ceil, kernelShape, attributes, x, y); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
