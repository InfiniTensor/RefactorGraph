#include "computation/operators/topk.h"
#include "kernel/collectors/topk.h"

namespace refactor::computation {

    size_t TopK::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t TopK::opTypeId() const noexcept { return typeId(); }
    std::string_view TopK::name() const noexcept { return "TopK"; }
    auto TopK::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::TopKCollector;
        return std::make_unique<Collector_>(target, topk, axis);
    }

}// namespace refactor::computation
