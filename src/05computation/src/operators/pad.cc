#include "computation/operators/pad.h"
#include "kernel/attributes/pad_info.h"

namespace refactor::computation {
    using Op = Pad;

    Op::Pad(decltype(pads) pads_,
            PadType mode_) noexcept : LayoutDependentOperator(), pads(std::move(pads_)), mode(mode_) {}

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "Pad"; }
    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::PadCollector;
        return std::make_unique<Collector_>(target, std::move(pads), mode);
    }
    auto Op::serialize() const noexcept -> std::string {
        return fmt::format("{}({}, {})",
                           name(),
                           vec2str(pads),
                           mode.toString());
    }

}// namespace refactor::computation
