#include "computation/operators/leaky_relu.h"

namespace refactor::computation {
    using Op = LeakyRelu;

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "LeakyRelu"; }

    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        return nullptr;
    }
    auto Op::serialize() const noexcept -> std::string {
        return fmt::format("{}({})", name(), alpha);
    }

}// namespace refactor::computation
