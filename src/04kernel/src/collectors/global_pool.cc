#include "kernel/collectors/global_pool.h"
#include "../kernels/pool/cudnn_kernel.hh"

namespace refactor::kernel {

    GlobalPoolCollector::GlobalPoolCollector(
        Target target_,
        PoolType type_) noexcept
        : InfoCollector(),
          target(target_),
          type(type_) {}

    std::vector<KernelBox>
    GlobalPoolCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &x = inputs[0].get();
        auto const &y = outputs[0].get();

        auto rank = x.rank();
        KernelShape kernelShape(rank - 2);
        std::transform(x.shape.begin() + 2, x.shape.end(),
                       kernelShape.begin(),
                       [](auto dim) { return static_cast<uint_lv1>(dim); });
        PoolAttributes attributes(rank - 2, nullptr, nullptr, nullptr);
        // TODO use pool kernels

        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                break;
            case Target::NvidiaGpu:
                if (auto ptr = PoolCudnn::build(type, false, kernelShape, attributes, x); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
