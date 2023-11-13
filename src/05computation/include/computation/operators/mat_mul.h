#ifndef COMPUTATION_MAT_MUL_H
#define COMPUTATION_MAT_MUL_H

#include "../operator.h"

namespace refactor::computation {

    struct MatMul final : public LayoutDependentOperator {
        float alpha, beta;
        bool transA, transB;

        constexpr MatMul(float alpha_, float beta_, bool transA_, bool transB_) noexcept
            : LayoutDependentOperator(),
              alpha(alpha_),
              beta(beta_),
              transA(transA_),
              transB(transB_) {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
    };

    using refactor::kernel::Tensor;
    struct MatMulBox final : public MyOperator {
        // Arc<MatMul> base;

        MatMulBox() noexcept : MyOperator() {
            base = std::make_shared<MatMul>(1.0, 1.0, false, false);
        }
        std::unique_ptr<Operator> clone() const final;
        bool compute(Tensor const &, Tensor const &, Tensor &) const noexcept final;
        Shape verify(Tensor const &, Tensor const &) const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_MAT_MUL_H
