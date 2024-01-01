#include "computation/operators/generator.h"

namespace refactor::computation {
    using Op = Generator;

    Op::Generator(decltype(subgraph) subgraph_)
        : Operator(), subgraph(std::move(subgraph_)) {}

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "Generator"; }
    auto Op::candidateKernels(Target target) const noexcept -> kernel::CollectorBox {
        TODO("");
    }

}// namespace refactor::computation
