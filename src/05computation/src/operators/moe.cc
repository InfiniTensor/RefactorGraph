#include "computation/operators/moe.h"
#include "kernel/collectors/moe.h"

namespace refactor::computation {

    auto AssignPos::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto AssignPos::opTypeId() const noexcept -> size_t { return typeId(); }
    auto AssignPos::name() const noexcept -> std::string_view { return "moe::AssignPos"; }
    auto AssignPos::candidateKernels(Target target) const -> kernel::CollectorBox {
        using Collector_ = kernel::AssignPosCollector;
        return std::make_unique<Collector_>(target, topk, numExperts);
    }
    auto AssignPos::serialize() const noexcept -> std::string {
        return "moe::AssignPos()";
    }

    auto Reorder::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Reorder::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Reorder::name() const noexcept -> std::string_view { return "moe::Reorder"; }
    auto Reorder::candidateKernels(Target target) const -> kernel::CollectorBox {
        using Collector_ = kernel::ReorderCollector;
        return std::make_unique<Collector_>(target, scatter, topk);
    }
    auto Reorder::serialize() const noexcept -> std::string {
        return "moe::Reorder()";
    }

}// namespace refactor::computation
