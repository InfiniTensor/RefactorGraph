#ifndef GRAPH_TOPO_LINKED_GRAPH_H
#define GRAPH_TOPO_LINKED_GRAPH_H

#include "common/range.h"
#include "container.h"
#include <algorithm>
#include <memory>
#include <sstream>
#include <unordered_map>

namespace refactor::graph_topo {
    using namespace common;

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

        Graph<TN, TE> toGraph(bool) const;

    public:
        LinkedGraph() = default;
        explicit LinkedGraph(Graph<TN, TE>);

        static auto shareEdge(TE) -> EdgeRc;

        std::string toString() const;
        Graph<TN, TE> toGraph() const;
        Graph<TN, TE> intoGraph() const;
        std::vector<NodeRc> const &nodes() const;
        std::vector<EdgeRc> const &inputs() const;
        std::vector<EdgeRc> const &outputs() const;
        void setInputs(std::vector<EdgeRc>);
        void setOutputs(std::vector<EdgeRc>);
        NodeRc pushNode(TN, std::vector<EdgeRc>);
        void eraseNode(size_t);
        void eraseNode(NodeRc);
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
        void connect(size_t i, EdgeRc);
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

    LINKED_GRAPH_FN toString() const->std::string {
        std::unordered_map<size_t, size_t> indices;
        std::stringstream ss;
        auto f = [&indices, &ss](EdgeRc const &e) {
            if (e) {
                auto [it, ok] = indices.try_emplace(reinterpret_cast<size_t>(e.get()), indices.size());
                ss << it->second << ' ';
            } else {
                ss << "? ";
            }
        };
        ss << "*. -> ( ";
        for (auto const &e : _inputs) { f(e); }
        ss << ')' << std::endl;
        for (auto i : range0_(_nodes.size())) {
            auto n = _nodes[i];
            ss << i << ". ( ";
            for (auto const &e : n->_inputs) { f(e); }
            ss << ") -> ( ";
            for (auto const &e : n->_outputs) { f(e); }
            ss << ')' << std::endl;
        }
        ss << "*. <- ( ";
        for (auto const &e : _outputs) { f(e); }
        ss << ')' << std::endl;
        return ss.str();
    }

    LINKED_GRAPH_FN nodes() const->std::vector<NodeRc> const & {
        return _nodes;
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

    LINKED_GRAPH_FN pushNode(TN info, std::vector<EdgeRc> outputs)->NodeRc {
        auto ans = Node::share(std::move(info), std::move(outputs));
        _nodes.push_back(ans);
        return ans;
    }

    LINKED_GRAPH_FN eraseNode(size_t i)->void {
        auto &node = _nodes.at(i);
        for (auto i : range0_(node->_inputs.size())) {
            node->disconnect(i);
        }
        for (auto i : range0_(node->_outputs.size())) {
            auto out = node->_outputs[i];
            while (!out->_targets.empty()) {
                auto target = out->_targets.begin()->first;
                for (auto j : range0_(target->_inputs.size())) {
                    if (target->_inputs[j] == out) {
                        target->disconnect(j);
                        break;
                    }
                }
            }
            for (auto j : range0_(_outputs.size())) {
                if (_outputs[j] == out) {
                    _outputs[j] = nullptr;
                }
            }
        }
        _nodes.erase(_nodes.begin() + i);
    }

    LINKED_GRAPH_FN eraseNode(NodeRc node)->void {
        for (auto i : range0_(_nodes.size())) {
            if (_nodes[i] == node) {
                eraseNode(i);
            }
        }
    }

    LINKED_GRAPH_FN Node::share(TN info, std::vector<EdgeRc> outputs)
        ->NodeRc {
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
        if (auto edge = std::exchange(_inputs.at(i), nullptr); edge) {
            auto it = edge->_targets.find(this->shared_from_this());
            if (0 == --it->second) {
                edge->_targets.erase(it);
            }
        }
        while (!_inputs.empty() && !_inputs.back()) {
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
            for (auto i : range0_(inputs.size())) { node->connect(i, edges[inputs[i]]); }
        }

        auto inputs = it.globalInputs();
        std::transform(inputs.begin(), inputs.end(), _inputs.begin(),
                       [&edges](auto i) { return edges[i]; });
        auto outputs = it.globalOutputs();
        _outputs.resize(outputs.size());
        std::transform(outputs.begin(), outputs.end(), _outputs.begin(),
                       [&edges](auto i) { return std::move(edges[i]); });
    }

    LINKED_GRAPH_FN toGraph(bool copy) const->Graph<TN, TE> {
        auto topology = GraphTopo::__withGlobalInputs(_inputs.size());
        std::vector<TN> nodes;
        std::vector<TE> edges;

        nodes.reserve(_nodes.size());
        edges.reserve(_inputs.size());

        std::unordered_map<size_t, size_t> edgeIndices;
        for (auto &e : _inputs) {
            edgeIndices.insert({reinterpret_cast<size_t>(e.get()), edges.size()});
            if (copy) {
                edges.push_back(e->_info);
            } else {
                edges.emplace_back(std::move(e->_info));
            }
        }

        for (auto &n : _nodes) {
            if (copy) {
                nodes.push_back(n->_info);
            } else {
                nodes.emplace_back(std::move(n->_info));
            }

            std::vector<size_t> newLocal, nodeInputs;
            nodeInputs.reserve(n->_inputs.size());
            for (auto &e : n->_inputs) {
                ASSERT(e, "Input edge is not connected");
                auto [it, ok] = edgeIndices.try_emplace(reinterpret_cast<size_t>(e.get()), edges.size());
                if (ok) {
                    ASSERT(!e->_source, "Local edge should not have source node");
                    newLocal.push_back(it->second);
                    if (copy) {
                        edges.push_back(e->_info);
                    } else {
                        edges.emplace_back(std::move(e->_info));
                    }
                }
                nodeInputs.push_back(it->second);
            }

            for (auto &e : n->_outputs) {
                edgeIndices.insert({reinterpret_cast<size_t>(e.get()), edges.size()});
                if (copy) {
                    edges.push_back(e->_info);
                } else {
                    edges.emplace_back(std::move(e->_info));
                }
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

    LINKED_GRAPH_FN toGraph() const->Graph<TN, TE> {
        return toGraph(true);
    }

    LINKED_GRAPH_FN intoGraph() const->Graph<TN, TE> {
        return toGraph(false);
    }

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_LINKED_GRAPH_H
