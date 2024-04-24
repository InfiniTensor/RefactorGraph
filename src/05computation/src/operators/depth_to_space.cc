#include "computation/operators/depth_to_space.h"

namespace refactor::computation {
    using Op = DepthToSpace;

    Op::DepthToSpace(decltype(blocksize) blocksize_, decltype(mode) mode_) noexcept
        : Operator(), blocksize(blocksize_), mode(mode_) {}

    size_t Op::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Op::opTypeId() const noexcept { return typeId(); }
    std::string_view Op::name() const noexcept { return "DepthToSpace"; }
    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::DepthToSpaceCollector;
        return std::make_unique<Collector_>(target, blocksize, mode);
    }
    auto Op::serialize() const noexcept -> std::string {
        return fmt::format("{}({}, {})", name(), blocksize, mode == DepthToSpaceMode::DCR ? "DCR" : "CRD");
    }

}// namespace refactor::computation
