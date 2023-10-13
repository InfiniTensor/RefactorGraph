#include "kernel/graph.h"

namespace refactor::kernel {

    Graph::Graph(Target target,
                 graph_topo::GraphTopo topology,
                 std::vector<_N> nodes,
                 std::vector<_E> edges) noexcept
        : _target(target),
          _internal(_G{
              std::move(topology),
              std::move(nodes),
              std::move(edges),
          }) {}

    runtime::Stream Graph::lower() const {
        std::vector<Routine> routines;
        std::vector<size_t> offsets;
        // TODO memory allocation
        return runtime::Stream(
            mem_manager::ForeignBlob::share(_target.memFunc(), 0),
            _internal.topology,
            std::move(routines),
            std::move(offsets));
    }

}// namespace refactor::kernel
