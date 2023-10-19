#include "computation/operators/gather.h"
#include "kernel/collectors/gather.h"

namespace refactor::computation {
    using Op = Gather;

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "Gather"; }
    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::GatherCollector;
        return std::make_unique<Collector_>(target, axis);
    }

}// namespace refactor::computation
