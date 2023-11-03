#include "computation/operators/mat_mul.h"
#include "kernel/collectors/mat_mul.h"

namespace refactor::computation {

    size_t MatMul::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t MatMul::opTypeId() const noexcept { return typeId(); }
    std::string_view MatMul::name() const noexcept { return "MatMul"; }
    auto MatMul::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        return std::make_unique<kernel::MatMulCollector>(target, alpha, beta, transA, transB);
    }
}// namespace refactor::computation
