#ifndef KERNEL_DEPTH_TO_SPACE_H
#define KERNEL_DEPTH_TO_SPACE_H

#include "../collector.h"

namespace refactor::kernel {

    enum class DepthToSpaceMode : uint8_t {
        DCR,
        CRD,
    };

    struct DepthToSpaceCollector final : public InfoCollector {
        uint32_t blocksize;
        DepthToSpaceMode mode;

        DepthToSpaceCollector(decltype(_target), decltype(blocksize), decltype(mode)) noexcept;

        std::vector<KernelBox>
        filter(TensorRefs inputs, TensorRefs outputs) const final;
    };

}// namespace refactor::kernel

#endif// KERNEL_DEPTH_TO_SPACE_H
