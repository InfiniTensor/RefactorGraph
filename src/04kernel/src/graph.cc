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
        auto const &g = _internal.contiguous();
        routines.reserve(g.nodes.size());
        std::transform(g.nodes.begin(), g.nodes.end(),
                       std::back_inserter(routines),
                       [](auto const &node) { return node.kernel->lower(); });
        auto [size, offsets] = flatAllocate(g);
        return runtime::Stream(
            mem_manager::ForeignBlob::share(_target.memFunc(), size),
            g.topology,
            std::move(routines),
            std::move(offsets));
    }

}// namespace refactor::kernel
