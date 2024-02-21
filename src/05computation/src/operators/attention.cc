#include "computation/operators/attention.h"
#include "kernel/collectors/attention.h"

namespace refactor::computation {
    using Op = Attention;

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "Attention"; }
    auto Op::candidateKernels(Target target) const -> kernel::CollectorBox {
        using Collector_ = kernel::AttentionCollector;
        return std::make_unique<Collector_>(target);
    }
    auto Op::serialize() const noexcept -> std::string {
        return "Attention()";
    }

}// namespace refactor::computation
