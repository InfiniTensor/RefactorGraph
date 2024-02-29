#include "computation/operators/rope.h"
#include "kernel/collectors/rope.h"

namespace refactor::computation {
    using Op = RotaryPositionEmbedding;

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "RotaryPositionEmbedding"; }
    auto Op::candidateKernels(Target target) const -> kernel::CollectorBox {
        using Collector_ = kernel::RoPECollector;
        return std::make_unique<Collector_>(target, theta);
    }
    auto Op::serialize() const noexcept -> std::string {
        union code {
            float f;
            int32_t i;
        };
        return fmt::format(("{}({:e}={:#010x})"),
                           name(), theta,
                           code{theta}.i);
    }
}// namespace refactor::computation
