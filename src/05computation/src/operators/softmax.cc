#include "computation/operators/softmax.h"
#include "kernel/collectors/softmax.h"

namespace refactor::computation {

    size_t Softmax::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Softmax::opTypeId() const noexcept { return typeId(); }
    std::string_view Softmax::name() const noexcept { return "Softmax"; }
    auto Softmax::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        using Collector_ = kernel::SoftmaxCollector;
        return std::make_unique<Collector_>(target, axis);
    }

}// namespace refactor::computation
