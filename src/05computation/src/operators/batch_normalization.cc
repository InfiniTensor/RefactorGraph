#include "computation/operators/batch_normalization.h"
#include "kernel/collectors/batch_normalization.h"

namespace refactor::computation {
    using Op = BatchNormalization;

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "BatchNormalization"; }
    auto Op::candidateKernels(Target target) const -> kernel::CollectorBox {
        using Collector_ = kernel::BatchNormalizationCollector;
        return std::make_unique<Collector_>(target, epsilon);
    }

}// namespace refactor::computation
