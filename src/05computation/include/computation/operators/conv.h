#ifndef COMPUTATION_CONV_H
#define COMPUTATION_CONV_H

#include "../operator.h"
#include "kernel/attributes/pool_attributes.h"

namespace refactor::computation {
    using kernel::PoolAttributes;

    struct Conv final : public Operator {
        PoolAttributes attributes;

        explicit Conv(PoolAttributes) noexcept;

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
    };

    using refactor::kernel::Tensor;
    struct ConvBox final : public MyOperator_B {
        // Arc<Conv> base;

        ConvBox() noexcept : MyOperator_B() {
            PoolAttributes attr(2, nullptr, nullptr, nullptr);
            base = std::make_shared<Conv>(attr);
        }
        bool compute(Tensor const &, Tensor const &, Tensor &) const noexcept {
            return false;
        }
        Shape verify(Tensor const &, Tensor const &) const noexcept {
            return {};
        }
    };

}// namespace refactor::computation

#endif// COMPUTATION_CONV_H
