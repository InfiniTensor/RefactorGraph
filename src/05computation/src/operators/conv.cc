#include "computation/operators/conv.h"
#include "kernel/collectors/conv.h"

namespace refactor::computation {
    using Op = Conv;

    Op::Conv(PoolAttributes attrs) noexcept
        : Operator(), attributes(std::move(attrs)) {}

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "Conv"; }
    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::ConvCollector;
        return std::make_unique<Collector_>(target, attributes);
    }
    auto Op::serialize() const noexcept -> std::string {
        return fmt::format("{}({})", name(), attributes.toString());
    }

}// namespace refactor::computation
