#include "computation/operators/mat_mul_integer.h"
#include "kernel/collectors/mat_mul_integer.h"

namespace refactor::computation {
    using Op = MatMulInteger;

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "MatMulInteger"; }
    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        return std::make_unique<kernel::MatMulIntegerCollector>(target);
    }
    auto Op::serialize() const noexcept -> std::string {
        return "MatMulInteger()";
    }

}// namespace refactor::computation
