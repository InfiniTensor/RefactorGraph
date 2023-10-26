#include "kernel/graph.h"

namespace refactor::kernel {

    Graph::Graph(Target target,
                 graph_topo::GraphTopo topology,
                 std::vector<_N> nodes,
                 std::vector<_E> edges) noexcept
        : _target(target),
          _internal(graph_topo::Graph<_N, _E>{
              std::move(topology),
              std::move(nodes),
              std::move(edges),
          }) {}

    runtime::Stream Graph::lower(Allocator allocator) const {
        std::vector<Routine> routines;
        routines.reserve(_internal.nodes.size());
        std::transform(_internal.nodes.begin(), _internal.nodes.end(),
                       std::back_inserter(routines),
                       [](auto const &node) {
                           return node.kernel
                                      ? node.kernel->lower()
                                      : refactor::runtime::emptyRoutine;
                       });
        auto [size, offsets] = allocator(_internal, sizeof(uint64_t));
        return runtime::Stream(
            mem_manager::ForeignBlob::share(_target.memManager(), size),
            _internal.topology,
            std::move(routines),
            std::move(offsets));
    }

}// namespace refactor::kernel
