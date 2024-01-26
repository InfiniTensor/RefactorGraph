#include "computation/operators/rms_normalization.h"
#include "kernel/collectors/rms_normalization.h"

namespace refactor::computation {
    using Op = RmsNormalization;

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "RmsNormalization"; }
    auto Op::candidateKernels(Target target) const -> kernel::CollectorBox {
        using Collector_ = kernel::RmsNormalizationCollector;
        return std::make_unique<Collector_>(target, epsilon);
    }
    auto Op::serialize() const noexcept -> std::string {
        union code {
            float f;
            int32_t i;
        };
        return fmt::format(("{}({:e}={:#010x})"),
                           name(), epsilon,
                           code{epsilon}.i);
    }

}// namespace refactor::computation
