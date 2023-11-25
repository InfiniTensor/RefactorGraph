#include "kernel/graph.h"
#include "runtime/mem_manager.hh"

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

        runtime::Resources res;
        res.fetchOrStore<runtime::MemManager>(_target.memManager());

        auto nodeCount = _internal.nodes.size();
        std::vector<size_t> workspace(nodeCount, 0);
        std::vector<runtime::Node> nodes;
        nodes.reserve(nodeCount);
        for (auto i : range0_(nodeCount)) {
            if (auto const &node = _internal.nodes[i]; node.kernel) {
                auto [routine, workspaceSize] = node.kernel->lower(res);
                nodes.emplace_back(routine);
                workspace[i] = workspaceSize;
            } else {
                nodes.emplace_back(runtime::emptyRoutine);
            }
        }

        auto [stack, edgeOffsets, worksapceOffsets] = allocator(_internal, workspace, 32);
        for (auto i : range0_(nodeCount)) {
            nodes[i].workspaceOffset = worksapceOffsets[i];
        }

        auto outputs = _internal.topology.globalOutputs();
        std::vector<size_t> outputs_(outputs.size());
        std::transform(outputs.begin(), outputs.end(),
                       outputs_.begin(),
                       [this](auto const &edge) { return _internal.edges[edge].size; });

        return runtime::Stream(
            std::move(res),
            stack,
            std::move(outputs_),
            _internal.topology,
            std::move(nodes),
            std::move(edgeOffsets));
    }

}// namespace refactor::kernel
