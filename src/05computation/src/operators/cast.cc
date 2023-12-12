#include "computation/operators/cast.h"
#include "kernel/collectors/cast.h"

namespace refactor::computation {
    using Op = Cast;

    size_t Op::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Op::opTypeId() const noexcept { return typeId(); }
    std::string_view Op::name() const noexcept { return "Cast"; }
    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::CastCollector;
        return std::make_unique<Collector_>(target);
    }
    auto Op::serialize() const noexcept -> std::string {
        return "Cast()";
    }

}// namespace refactor::computation
