#include "computation/graph.h"
#include <numeric>

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

    std::pair<std::string, std::vector<uint8_t>> Graph::serialize() const {
        auto const &graph = _internal.contiguous();
        graph_topo::LinkedGraph<count_t, count_t> temp;
        {
            std::vector<count_t>
                nodes(graph.nodes.size()),
                edges(graph.edges.size());
            std::iota(nodes.begin(), nodes.end(), 0);
            std::iota(edges.begin(), edges.end(), 0);
            std::unordered_map<count_t, count_t> identities;

            for (auto [nodeIdx, inputs, outputs] : graph.topology) {
                if (auto const &op = graph.nodes[nodeIdx].op; op && op->isIdentity()) {
                    identities.emplace(outputs[0], inputs[0]);
                }
            }

            auto modifier = graph_topo::InplaceModifier(graph.topology);
            modifier.reconnect(identities);

            temp = graph_topo::LinkedGraph(graph_topo::Graph{
                modifier.take(),
                std::move(nodes),
                std::move(edges),
            });
            temp.cleanup();
            for (auto const &node : temp.nodes()) {
                auto const &inputs = node->inputs();
                for (auto i : range0_(inputs.size()).rev()) {
                    if (!graph.edges[inputs[i]->info()].tensor) {
                        node->disconnect(i);
                    } else {
                        break;
                    }
                }
            }
        }

        std::unordered_map<Tensor *, count_t> edgeFlags;
        std::vector<std::string> edges;

        auto addEdge = [&](Rc<decltype(temp)::Edge> const &edge) {
            auto const &[tensor, name] = graph.edges[edge->info()];
            ASSERT(tensor, "");
            auto [it, ok] = edgeFlags.try_emplace(tensor.get(), static_cast<count_t>(edges.size()));
            if (ok) {
                edges.push_back(fmt::format(
                    "{:>5}.\t{:<32}\t{:<12}\t{}",
                    it->second,
                    name,
                    tensor->dataType.name(),
                    vec2str(tensor->shape)));
            }
            return it->second;
        };
        auto getEdge = [&](Rc<decltype(temp)::Edge> const &edge) {
            return edgeFlags.at(graph.edges[edge->info()].tensor.get());
        };

        for (auto const &edge : temp.inputs()) {
            addEdge(edge);
        }

        std::stringstream ss;
        for (auto i = 0; auto node : temp.nodes()) {
            auto const &[op, name] = graph.nodes[node->info()];
            if (auto i_ = ++i; op) {
                ss << fmt::format("{:>5}.\t{:<32}\t{:<16}", i_, name, op->name());
            }

            for (auto const &edge : node->outputs()) {
                ss << ' ' << addEdge(edge);
            }
            ss << " <-";
            for (auto const &edge : node->inputs()) {
                ss << ' ' << addEdge(edge);
            }
            ss << std::endl;
        }
        ss << std::endl;
        {
            ss << "graph.";
            for (auto const &edge : temp.outputs()) {
                ss << ' ' << getEdge(edge);
            }
            ss << " <-";
            for (auto const &edge : temp.inputs()) {
                ss << ' ' << getEdge(edge);
            }
            ss << std::endl;
        }
        ss << std::endl;
        for (auto const &edge : edges) {
            ss << edge << std::endl;
        }

        std::vector<uint8_t> data;
        return {ss.str(), std::move(data)};
    }

}// namespace refactor::computation
