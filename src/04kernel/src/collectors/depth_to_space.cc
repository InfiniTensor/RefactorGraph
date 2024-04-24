#include "kernel/collectors/depth_to_space.h"
#include "../kernels/transpose/cpu_kernel.hh"
#include "../kernels/transpose/cuda_kernel.hh"

namespace refactor::kernel {

    DepthToSpaceCollector::DepthToSpaceCollector(
        decltype(_target) target,
        decltype(blocksize) blocksize_,
        decltype(mode) mode_) noexcept
        : InfoCollector(target),
          blocksize(blocksize_),
          mode(mode_) {}

    std::vector<KernelBox>
    DepthToSpaceCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &input = inputs[0].get();
        auto const n = input.shape[0];
        auto const c = input.shape[1];
        auto const h = input.shape[2];
        auto const w = input.shape[3];
        auto info = mode == DepthToSpaceMode::DCR
                        ? TransposeInfo(
                              input.dataType,
                              {n, blocksize, blocksize, c / blocksize / blocksize, h, w},
                              {0, 3, 4, 1, 5, 2})
                        : TransposeInfo(
                              input.dataType,
                              {n, c / blocksize / blocksize, blocksize, blocksize, h, w},
                              {0, 1, 4, 2, 5, 3});

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = TransposeCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = TransposeCuda::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
