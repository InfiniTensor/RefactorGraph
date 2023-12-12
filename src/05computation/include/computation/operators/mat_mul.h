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
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_MAT_MUL_H
