#ifndef COMPUTATION_MOE_H
#define COMPUTATION_MOE_H

#include "../operator.h"

namespace refactor::computation {

    struct AssignPos final : public Operator {
        uint32_t topk,numExperts;

        constexpr explicit AssignPos(uint32_t topk, uint32_t numExperts) noexcept : Operator(), 
            topk(topk), numExperts(numExperts){}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const final;
        std::string serialize() const noexcept final;
    };

     struct Reorder final : public Operator {
        bool scatter;
        uint32_t topk;

        constexpr explicit Reorder(bool scatter, uint32_t topk) noexcept : Operator(), 
            scatter(scatter), topk(topk){}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_RMS_NORMALIZATION_H
