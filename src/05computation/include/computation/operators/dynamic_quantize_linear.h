#ifndef COMPUTATION_DYNAMIC_QUANTIZE_LINEAR_H
#define COMPUTATION_DYNAMIC_QUANTIZE_LINEAR_H

#include "../operator.h"

namespace refactor::computation {

    struct DynamicQuantizeLinear final : public Operator {

        constexpr DynamicQuantizeLinear() noexcept = default;

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_DYNAMIC_QUANTIZE_LINEAR_H
