#ifndef GRAPH_TOPO_LINKED_GRAPH_H
#define GRAPH_TOPO_LINKED_GRAPH_H

#include "container.h"
#include <algorithm>
#include <memory>
#include <unordered_map>

namespace refactor::graph_topo {

    template<class TN, class TE>
    class LinkedGraph {
    public:
        class Node;
        class Edge;

    private:
        using NodeRc = std::shared_ptr<Node>;
        using EdgeRc = std::shared_ptr<Edge>;

        std::vector<NodeRc> _nodes;
        std::vector<EdgeRc> _inputs, _outputs;

    public:
        LinkedGraph() = default;
        explicit LinkedGraph(Graph<TN, TE>);

        static auto shareEdge(TE) -> EdgeRc;

        Graph<TN, TE> build() const;
        std::vector<EdgeRc> const &inputs() const;
        std::vector<EdgeRc> const &outputs() const;
        void setInputs(std::vector<EdgeRc>);
        void setOutputs(std::vector<EdgeRc>);
        NodeRc addNode(TN, std::vector<EdgeRc>);
    };

    template<class TN, class TE>
    class LinkedGraph<TN, TE>::Node : public std::enable_shared_from_this<LinkedGraph<TN, TE>::Node> {
        friend class LinkedGraph<TN, TE>;
        friend class Edge;

        TN _info;
        std::vector<EdgeRc> _inputs, _outputs;

        Node(TN, std::vector<EdgeRc>);
        static NodeRc share(TN, std::vector<EdgeRc>);

    public:
        TN const &info() const;
        std::vector<EdgeRc> const &inputs() const;
        std::vector<EdgeRc> const &outputs() const;
        void connect(size_t i, EdgeRc input);
        void disconnect(size_t i);
    };

    template<class TN, class TE>
    class LinkedGraph<TN, TE>::Edge : public std::enable_shared_from_this<LinkedGraph<TN, TE>::Edge> {
        friend class LinkedGraph<TN, TE>;
        friend class Edge;

        TE _info;
        NodeRc _source;
        std::unordered_map<NodeRc, size_t> _targets;

    public:
        explicit Edge(TE);
        TE const &info() const;
    };

#define LINKED_GRAPH_FN template<class TN, class TE> auto LinkedGraph<TN, TE>::
#define LINKED_GRAPH_CONSTRUCTOR template<class TN, class TE> LinkedGraph<TN, TE>::

    LINKED_GRAPH_FN shareEdge(TE info)->EdgeRc {
        return std::make_shared<Edge>(std::move(info));
    }

    LINKED_GRAPH_FN inputs() const->std::vector<EdgeRc> const & {
        return _inputs;
    }

    LINKED_GRAPH_FN outputs() const->std::vector<EdgeRc> const & {
        return _outputs;
    }

    LINKED_GRAPH_FN setInputs(std::vector<EdgeRc> inputs)->void {
        _inputs = std::move(inputs);
    }

    LINKED_GRAPH_FN setOutputs(std::vector<EdgeRc> outputs)->void {
        _outputs = std::move(outputs);
    }

    LINKED_GRAPH_FN addNode(TN info, std::vector<EdgeRc> outputs)->NodeRc {
        auto ans = Node::share(std::move(info), std::move(outputs));
        _nodes.push_back(ans);
        return ans;
    }

    LINKED_GRAPH_FN Node::share(TN info, std::vector<EdgeRc> outputs)->NodeRc {
        auto ans = std::shared_ptr<Node>(new Node(std::move(info), std::move(outputs)));
        for (auto &edge : ans->_outputs) {
            edge->_source = ans;
        }
        return ans;
    }

    LINKED_GRAPH_FN Node::info() const->TN const & {
        return _info;
    }

    LINKED_GRAPH_FN Node::inputs() const->std::vector<EdgeRc> const & {
        return _inputs;
    }

    LINKED_GRAPH_FN Node::outputs() const->std::vector<EdgeRc> const & {
        return _outputs;
    }

    LINKED_GRAPH_FN Node::connect(size_t i, EdgeRc input)->void {
        if (i < _inputs.size()) {
            disconnect(i);
        } else {
            _inputs.resize(i + 1, nullptr);
        }
        if (input) {
            ++input->_targets.try_emplace(this->shared_from_this(), 0).first->second;
            _inputs.at(i) = std::move(input);
        }
    }

    LINKED_GRAPH_FN Node::disconnect(size_t i)->void {
        auto edge = std::exchange(_inputs.at(i), nullptr);
        if (edge) {
            auto it = edge->_targets.find(this->shared_from_this());
            if (!--it->second) {
                edge->_targets.erase(it);
            }
        }
        while (!_inputs.back()) {
            _inputs.pop_back();
        }
    }

    LINKED_GRAPH_FN Edge::info() const->TE const & {
        return _info;
    }

    LINKED_GRAPH_CONSTRUCTOR Node::Node(TN info, std::vector<EdgeRc> outputs)
        : _info(std::move(info)),
          _inputs(),
          _outputs(std::move(outputs)) {}

    LINKED_GRAPH_CONSTRUCTOR Edge::Edge(TE info)
        : _info(std::move(info)) {}

    LINKED_GRAPH_CONSTRUCTOR LinkedGraph(Graph<TN, TE> g)
        : _inputs(g.topology.globalInputsCount()),
          _outputs(),
          _nodes(g.topology.nodeCount()) {

        std::vector<EdgeRc> edges(g.edges.size());
        std::transform(g.edges.begin(), g.edges.end(), edges.begin(),
                       [](auto &e) { return shareEdge(std::move(e)); });

        auto it = g.topology.begin(), end_ = g.topology.end();
        while (it != end_) {
            auto [nodeIdx, inputs, outputs] = *it++;
            std::vector<EdgeRc> outputs_(outputs.size());
            std::transform(outputs.begin(), outputs.end(), outputs_.begin(),
                           [&edges](auto i) { return edges[i]; });
            auto &node = _nodes[nodeIdx] = Node::share(std::move(g.nodes[nodeIdx]), std::move(outputs_));
            for (auto i : inputs) { node->connect(i, edges[i]); }
        }

        auto inputs = it.globalInputs();
        std::transform(inputs.begin(), inputs.end(), _inputs.begin(),
                       [&edges](auto i) { return edges[i]; });
        auto outputs = it.globalOutputs();
        _outputs.resize(outputs.size());
        std::transform(outputs.begin(), outputs.end(), _outputs.begin(),
                       [&edges](auto i) { return std::move(edges[i]); });
    }

    LINKED_GRAPH_FN build() const->Graph<TN, TE> {
        auto topology = GraphTopo::__withGlobalInputs(_inputs.size());
        std::vector<TN> nodes;
        std::vector<TE> edges;

        nodes.reserve(_nodes.size());
        edges.reserve(_inputs.size());

        std::unordered_map<size_t, size_t> edgeIndices;
        for (auto &e : _inputs) {
            edgeIndices.insert({reinterpret_cast<size_t>(e.get()), edges.size()});
            edges.push_back(std::move(e->_info));
        }

        for (auto &n : _nodes) {
            nodes.push_back(std::move(n->_info));

            std::vector<size_t> newLocal, nodeInputs;
            nodeInputs.reserve(n->_inputs.size());
            for (auto &e : n->_inputs) {
                auto [it, ok] = edgeIndices.try_emplace(reinterpret_cast<size_t>(e.get()), edges.size());
                if (ok) {
                    ASSERT(!e->_source, "Local edge should not have source node");
                    newLocal.push_back(it->second);
                    edges.push_back(std::move(e->_info));
                }
                nodeInputs.push_back(it->second);
            }

            for (auto &e : n->_outputs) {
                edgeIndices.insert({reinterpret_cast<size_t>(e.get()), edges.size()});
                edges.push_back(std::move(e->_info));
            }

            topology.__addNode(newLocal.size(), std::move(nodeInputs), n->_outputs.size());
        }

        std::vector<size_t> outputs(_outputs.size());
        std::transform(_outputs.begin(), _outputs.end(), outputs.begin(),
                       [&](auto &e) { return edgeIndices.at(reinterpret_cast<size_t>(e.get())); });
        topology.__setGlobalOutputs(std::move(outputs));

        return {
            std::move(topology),
            std::move(nodes),
            std::move(edges),
        };
    }

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_LINKED_GRAPH_H
