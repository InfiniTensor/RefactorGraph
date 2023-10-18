#include "computation/graph.h"
#include "refactor/common.h"

namespace refactor::computation {

    Graph::Graph(graph_topo::Graph<Node, Edge> internal) noexcept
        : _internal(std::move(internal)) {}
    Graph::Graph(graph_topo::GraphTopo topology,
                 std::vector<Node> nodes,
                 std::vector<Edge> edges) noexcept
        : Graph(graph_topo::Graph<Node, Edge>{
              std::move(topology),
              std::move(nodes),
              std::move(edges),
          }) {}

    kernel::Graph Graph::lower(kernel::Target target) const {
        auto const &graph = _internal.contiguous();

        std::vector<kernel::Node> nodes(graph.nodes.size());
        std::vector<kernel::Edge> edges(graph.edges.size());

        for (auto [nodeIdx, inputs, outputs] : graph.topology) {
            auto const &[op, name] = graph.nodes[nodeIdx];
            kernel::TensorRefs inputs_, outputs_;
            inputs_.reserve(inputs.size());
            outputs_.reserve(outputs.size());
            std::transform(inputs.begin(), inputs.end(),
                           std::back_inserter(inputs_), [&](auto i) {
                               return std::cref(*graph.edges[i].tensor);
                           });
            std::transform(outputs.begin(), outputs.end(),
                           std::back_inserter(outputs_), [&](auto i) {
                               return std::cref(*graph.edges[i].tensor);
                           });
            auto candidates = op->candidateKernels(target)->filter(std::move(inputs_), std::move(outputs_));
            ASSERT(!candidates.empty(), "No kernel selected");
            nodes[nodeIdx] = {std::move(candidates.front()), name};
        }

        for (auto i : range0_(edges.size())) {
            auto const &[tensor, name] = graph.edges[i];
            if (!tensor || !tensor->data) { continue; }
            auto fn = target.memFunc();
            auto blob = mem_manager::ForeignBlob::share(fn, tensor->bytesSize());
            fn.copyHd(*blob, *(tensor->data), tensor->bytesSize());
            edges[i] = {std::move(blob), tensor->bytesSize(), name};
        }

        return kernel::Graph(target, graph.topology, std::move(nodes), std::move(edges));
    }

}// namespace refactor::computation
