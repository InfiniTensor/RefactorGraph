#ifndef COMPUTATION_WHERE_H
#define COMPUTATION_WHERE_H

#include "../operator.h"

namespace refactor::computation {

    struct Where final : public Operator {
        constexpr Where() noexcept : Operator() {}

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_WHERE_H
