#include "kernel/collectors/slice.h"
#include "../kernels/slice/cpu_kernel.hh"
#include "../kernels/slice/cuda_kernel.hh"
#include "../kernels/slice/cnnl_kernel.hh"

namespace refactor::kernel {

    SliceCollector::SliceCollector(
        decltype(_target) target,
        Dimensions dims) noexcept
        : InfoCollector(target),
          dimentions(std::move(dims)) {}

    std::vector<KernelBox>
    SliceCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        SliceInfo info(dimentions, inputs[0]);

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = SliceCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = SliceCuda::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Mlu:
                if (auto ptr = SliceCnnl::build(inputs[0].get().dataType, dimentions, inputs[0].get().shape, outputs[0].get().shape); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
