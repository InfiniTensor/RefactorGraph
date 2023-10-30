#include "computation/operators/where.h"
#include "kernel/collectors/where.h"

namespace refactor::computation {

    size_t Where::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Where::opTypeId() const noexcept { return typeId(); }
    std::string_view Where::name() const noexcept { return "Where"; }
    auto Where::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::WhereCollector;
        return std::make_unique<Collector_>(target);
    }

}// namespace refactor::computation
