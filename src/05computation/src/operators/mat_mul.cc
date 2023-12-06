#include "computation/operators/mat_mul.h"
#include "kernel/collectors/mat_mul.h"

namespace refactor::computation {
    using Op = MatMul;

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "MatMul"; }
    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        return std::make_unique<kernel::MatMulCollector>(target, alpha, beta, transA, transB);
    }
    auto Op::serialize() const noexcept -> std::string {
        union code {
            float f;
            int32_t i;
        };

        return fmt::format("{}({:e}={:#010x}, {:e}={:#010x}, A{}, B{})",
                           name(),
                           alpha, code{alpha}.i,
                           beta, code{beta}.i,
                           transA ? "T" : "",
                           transB ? "T" : "");
    }

}// namespace refactor::computation
