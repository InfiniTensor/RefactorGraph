#ifndef COMPUTATION_CONCAT_H
#define COMPUTATION_CONCAT_H

#include "../operator.h"

namespace refactor::computation {

    struct Concat final : public AxisRankOperator {
        constexpr Concat(uint32_t axis, uint32_t rank) noexcept
            : AxisRankOperator(axis, rank) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
    };

    using refactor::kernel::Tensor;
    struct ConcatBox final : public MyOperator_B {
        // Arc<Concat> base;

        ConcatBox() noexcept : MyOperator_B() {
            base = std::make_shared<Concat>(1, 2);
        }
        bool compute(Tensor const &, Tensor const &, Tensor &) const noexcept final;
        Shape verify(Tensor const &, Tensor const &) const noexcept final;
    };
}// namespace refactor::computation

#endif// COMPUTATION_CONCAT_H
