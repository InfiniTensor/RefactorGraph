#include "runtime/stream.h"

namespace refactor::runtime {

    Stream::Stream(mem_manager::SharedForeignBlob stack,
                   graph_topo::GraphTopo topology,
                   std::vector<_N> routines,
                   std::vector<_E> offsets)
        : _stack(std::move(stack)),
          _internal(_G{
              std::move(topology),
              std::move(routines),
              std::move(offsets),
          }) {}

    void Stream::run() {
        auto stack = _stack->ptr();
        for (auto [nodeIdx, inputs, outputs] : _internal.topology) {
            auto const &routine = _internal.nodes[nodeIdx];
            Addresses
                inputs_(inputs.size()),
                outputs_(outputs.size());
            std::transform(inputs.begin(), inputs.end(),
                           inputs_.begin(),
                           [stack, this](auto i) { return stack + _internal.edges[i]; });
            std::transform(outputs.begin(), outputs.end(),
                           outputs_.begin(),
                           [stack, this](auto i) { return stack + _internal.edges[i]; });
            routine(_resources, std::move(inputs_), std::move(outputs_));
        }
    }

}// namespace refactor::runtime
