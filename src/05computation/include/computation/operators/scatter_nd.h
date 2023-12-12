#ifndef COMPUTATION_SCATTER_ND_H
#define COMPUTATION_SCATTER_ND_H

#include "../operator.h"

namespace refactor::computation {

    struct ScatterND final : public Operator {

        ScatterND() = default;

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
        std::string serialize() const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_SCATTER_ND_H
