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

        auto cleaner = graph_topo::LinkedGraph(graph_topo::Graph{
            modifier.take(),
            std::move(nodes),
            std::move(edges),
        });
        cleaner.cleanup();
        auto [topo__, nodes__, edges__] = cleaner.intoGraph();

        return kernel::Graph(std::move(topo__), std::move(nodes__), std::move(edges__));
    }

    auto Graph::internal() const -> decltype(_internal) const & { return _internal; }

    class EdgeRecorder {
        std::unordered_map<Tensor *, count_t> _indices;
        std::vector<std::string> _edges;
        std::vector<uint8_t> _data;
        bool _extractData;

    public:
        friend std::ostream &operator<<(std::ostream &os, EdgeRecorder const &edges);

        explicit EdgeRecorder(bool extractData) noexcept
            : _extractData(extractData) {}

        count_t operator+=(Edge const &e) {
            auto const &[tensor, name] = e;
            ASSERT(tensor, "");
            auto [it, ok] = _indices.try_emplace(tensor.get(), static_cast<count_t>(_edges.size()));
            if (ok) {
                std::string data;
                if (_extractData && tensor->data) {
                    auto ptr = tensor->data->get<uint8_t>();
                    auto len = tensor->bytesSize();
                    data = fmt::format("{:>#10x} +{:<#10x}", _data.size(), len);
                    _data.insert(_data.end(), ptr, ptr + len);
                } else {
                    data = fmt::format("{:>#10x} +{:<#10x}", 0, 0);
                }
                _edges.push_back(fmt::format(
                    "{:>7}\t{:<32}\t{:<12}\t{}\t{}\t{}",
                    fmt::format("%{}.", it->second),
                    name,
                    tensor->dataType.name(),
                    tensor->layout == LayoutType::NCHW   ? "NCHW"
                    : tensor->layout == LayoutType::NHWC ? "NHWC"
                                                         : "ELSE",
                    data,
                    vec2str(tensor->shape)));
            }
            return it->second;
        }
        count_t operator[](Edge const &e) const {
            return _indices.at(e.tensor.get());
        }
        decltype(_data) takeData() {
            return std::move(_data);
        }
    };
    std::ostream &operator<<(std::ostream &os, EdgeRecorder const &edges) {
        for (auto const &edge : edges._edges) {
            os << edge << std::endl;
        }
        return os;
    }

    std::pair<std::string, std::vector<uint8_t>> Graph::serialize(bool withData) const {
        auto const &graph = _internal.contiguous();
        graph_topo::LinkedGraph<count_t, count_t> cleaner;
        {
            std::unordered_map<count_t, count_t> identities;
            for (auto [nodeIdx, inputs, outputs] : graph.topology) {
                if (auto const &op = graph.nodes[nodeIdx].op; op && op->isIdentity()) {
                    identities.emplace(outputs[0], inputs[0]);
                }
            }

            auto modifier = graph_topo::InplaceModifier(graph.topology);
            modifier.reconnect(identities);

            std::vector<count_t>
                nodes(graph.nodes.size()),
                edges(graph.edges.size());
            std::iota(nodes.begin(), nodes.end(), 0);
            std::iota(edges.begin(), edges.end(), 0);

            cleaner = graph_topo::LinkedGraph(graph_topo::Graph{
                modifier.take(),
                std::move(nodes),
                std::move(edges),
            });
            cleaner.cleanup();

            for (auto const &n : cleaner.nodes()) {
                auto const &inputs = n->inputs();
                for (auto i : range0_(inputs.size()).rev()) {
                    if (!graph.edges[inputs[i]->info()].tensor) {
                        n->disconnect(i);
                    } else {
                        break;
                    }
                }
            }
        }

        EdgeRecorder edges(withData);
        for (auto const &edge : cleaner.inputs()) {
            edges += graph.edges[edge->info()];
        }
        std::stringstream ss;
        for (auto i = 0; auto n : cleaner.nodes()) {
            auto const &[op, name] = graph.nodes[n->info()];
            if (auto i_ = ++i; op) {
                ss << fmt::format("{:>5}.\t{:<32}\t{}", i_, name, op->serialize());
            }

            for (auto const &e : n->outputs()) {
                ss << " %" << (edges += graph.edges[e->info()]);
            }
            ss << " <-";
            for (auto const &e : n->inputs()) {
                ss << " %" << (edges += graph.edges[e->info()]);
            }
            ss << std::endl;
        }
        ss << std::endl
           << "graph.";
        for (auto const &e : cleaner.outputs()) {
            ss << " %" << edges[graph.edges[e->info()]];
        }
        ss << " <-";
        for (auto const &e : cleaner.inputs()) {
            ss << " %" << edges[graph.edges[e->info()]];
        }
        ss << std::endl
           << std::endl
           << edges;

        return {ss.str(), edges.takeData()};
    }

}// namespace refactor::computation
