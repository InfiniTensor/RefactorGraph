#include "computation/operators/hard_sigmoid.h"
#include "kernel/collectors/hard_sigmoid.h"

namespace refactor::computation {
    using Op = HardSigmoid;

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "HardSigmoid"; }

    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::HardSigmoidCollector;
        return std::make_unique<Collector_>(target, alpha, beta);
    }
    auto Op::serialize() const noexcept -> std::string {
        return fmt::format("{}()", name());
    }

}// namespace refactor::computation

