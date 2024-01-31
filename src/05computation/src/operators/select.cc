#include "computation/operators/select.h"
#include "kernel/collectors/select.h"

namespace refactor::computation {

    size_t Select::typeId(SelectType type) noexcept {
        switch (type) {
            case SelectType::Min: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case SelectType::Max: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }
    size_t Select::opTypeId() const noexcept {
        return typeId(type);
    }
    std::string_view Select::name() const noexcept {
        switch (type) {
            case SelectType::Min:
                return "Min";
            case SelectType::Max:
                return "Max";
            default:
                UNREACHABLE();
        }
    }

    auto Select::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::SelectCollector;
        return std::make_unique<Collector_>(target, type);
    }

}// namespace refactor::computation
