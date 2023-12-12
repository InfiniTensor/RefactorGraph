#include "computation/operators/scatter_nd.h"
#include "kernel/collectors/scatter_nd.h"

namespace refactor::computation {
    using Op = ScatterND;

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "ScatterND"; }
    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::ScatterNDCollector;
        return std::make_unique<Collector_>(target);
    }
    auto Op::serialize() const noexcept -> std::string {
        return "ScatterND()";
    }

}// namespace refactor::computation
