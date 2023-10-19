#include "kernel/graph.h"
#include "allocator/flat_allocator.h"

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

    runtime::Stream Graph::lower() const {
        std::vector<Routine> routines;
        routines.reserve(_internal.nodes.size());
        std::transform(_internal.nodes.begin(), _internal.nodes.end(),
                       std::back_inserter(routines),
                       [](auto const &node) { return node.kernel->lower(); });
        auto [size, offsets] = flatAllocate(_internal, 64);
        return runtime::Stream(
            mem_manager::ForeignBlob::share(_target.memFunc(), size),
            _internal.topology,
            std::move(routines),
            std::move(offsets));
    }

}// namespace refactor::kernel
