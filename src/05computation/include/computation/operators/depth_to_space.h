#ifndef COMPUTATION_DEPTH_TO_SPACE_H
#define COMPUTATION_DEPTH_TO_SPACE_H

#include "../operator.h"
#include "kernel/collectors/depth_to_space.h"

namespace refactor::computation {
    using kernel::DepthToSpaceMode;

    struct DepthToSpace final : public Operator {
        uint32_t blocksize;
        DepthToSpaceMode mode;

        DepthToSpace(decltype(blocksize), decltype(mode)) noexcept;

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_DEPTH_TO_SPACE_H
