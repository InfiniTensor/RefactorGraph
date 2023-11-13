#include "kernel/collectors/slice.h"
#include "../kernels/slice/cpu_kernel.hh"
#include "../kernels/slice/cuda_kernel.hh"

namespace refactor::kernel {

    SliceCollector::SliceCollector(
        Target target_,
        Dimensions dims) noexcept
        : InfoCollector(),
          target(target_),
          dimentions(std::move(dims)) {}

    std::vector<KernelBox>
    SliceCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        SliceInfo info(dimentions, inputs[0]);

        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                if (auto ptr = SliceCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case Target::NvidiaGpu:
                if (auto ptr = SliceCuda::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
