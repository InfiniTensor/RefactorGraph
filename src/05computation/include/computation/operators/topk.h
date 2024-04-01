#ifndef COMPUTATION_TOPK_H
#define COMPUTATION_TOPK_H

#include "../operator.h"

namespace refactor::computation {

    struct TopK final : public Operator {
        int64_t topk, axis;
        constexpr TopK(int64_t topk, int64_t axis) noexcept : topk(topk), axis(axis){}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SPLIT_H
