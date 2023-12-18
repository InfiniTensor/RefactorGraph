#ifndef COMPUTATION_DEQUANTIZE_LINEAR_H
#define COMPUTATION_DEQUANTIZE_LINEAR_H

#include "../operator.h"

namespace refactor::computation {

    struct DequantizeLinear final : public Operator {

        constexpr DequantizeLinear() noexcept = default;

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_DEQUANTIZE_LINEAR_H
