#ifndef COMPUTATION_GENERATOR_H
#define COMPUTATION_GENERATOR_H

#include "../graph.h"
#include "../operator.h"

namespace refactor::computation {

    struct Generator final : public Operator {

        Graph subgraph;

        explicit Generator(decltype(subgraph));

        static size_t typeId() noexcept;
        size_t opTypeId() const noexcept final;
        std::string_view name() const noexcept final;
        kernel::CollectorBox candidateKernels(Target) const noexcept final;
    };

}// namespace refactor::computation

#endif// COMPUTATION_GENERATOR_H
