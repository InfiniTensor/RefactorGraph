#include "computation/graph.h"

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

    kernel::Graph Graph::lower(Target target) const {
        auto const &graph = _internal.contiguous();

        std::vector<kernel::Node> nodes(graph.nodes.size());
        std::vector<kernel::Edge> edges(graph.edges.size());

        std::unordered_map<count_t, count_t> identities;
        for (auto [nodeIdx, inputs, outputs] : graph.topology) {
            auto const &[op, name] = graph.nodes[nodeIdx];
            nodes[nodeIdx] = {nullptr, name};
            if (!op) {
                continue;
            }
            if (op->isIdentity()) {
                auto [it, ok] = identities.try_emplace(outputs[0], inputs[0]);
                ASSERT(ok, "");
                continue;
            }
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
            ASSERT(!candidates.empty(), "No kernel selected for \"{}\"", name);
            nodes[nodeIdx].kernel = std::move(candidates.front());
        }

        for (auto i : range0_(edges.size())) {
            auto const &[tensor, name] = graph.edges[i];
            if (!tensor || identities.contains(i)) {
                edges[i] = {nullptr, 0, name};
            } else {
                edges[i] = {tensor->data, tensor->bytesSize(), name};
            }
        }

        auto modifier = graph_topo::InplaceModifier(graph.topology);
        modifier.reconnect(identities);

        auto temp = graph_topo::LinkedGraph(graph_topo::Graph{
            modifier.take(),
            std::move(nodes),
            std::move(edges),
        });
        temp.cleanup();
        auto [topo__, nodes__, edges__] = temp.intoGraph();

        return kernel::Graph(std::move(topo__), std::move(nodes__), std::move(edges__));
    }

    auto Graph::internal() const -> decltype(_internal) const & { return _internal; }

}// namespace refactor::computation
