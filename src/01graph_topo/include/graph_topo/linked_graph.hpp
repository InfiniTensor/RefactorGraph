#ifndef GRAPH_TOPO_LINKED_GRAPH_H
#define GRAPH_TOPO_LINKED_GRAPH_H

#include "container.h"
#include <unordered_set>

namespace refactor::graph_topo {

    template<class TN, class TE>
    class LinkedGraph {
    public:
        class Node;
        class Edge;

    private:
        std::vector<Rc<Node>> _nodes;
        std::vector<Rc<Edge>> _inputs, _outputs;

        // 清理一个节点的输入输出但不移动节点。
        void _cleanupNode(Node &);

    public:
        LinkedGraph() = default;
        explicit LinkedGraph(Graph<TN, TE>);

        static auto shareEdge(TE) -> Rc<Edge>;

        std::string toString() const;
        Graph<TN, TE> intoGraph() const;
        std::vector<Rc<Node>> const &nodes() const;
        std::vector<Rc<Edge>> const &inputs() const;
        std::vector<Rc<Edge>> const &outputs() const;
        void setInputs(std::vector<Rc<Edge>>);
        void setOutputs(std::vector<Rc<Edge>>);
        void replaceInput(Rc<Edge>, Rc<Edge>);
        void replaceOutput(Rc<Edge>, Rc<Edge>);
        Rc<Node> pushNode(TN, std::vector<Rc<Edge>>);
        void eraseNode(count_t);
        void eraseNode(Rc<Node>);
        size_t cleanup(bool useless(TE const &) = nullptr);
        bool sort();
    };

    template<class TN, class TE>
    class LinkedGraph<TN, TE>::Node : public RefInplace<LinkedGraph<TN, TE>::Node> {
        friend class LinkedGraph<TN, TE>;
        friend class Edge;

        TN _info;
        std::vector<Rc<Edge>> _inputs, _outputs;

        Node(TN, std::vector<Rc<Edge>>);
        static Rc<Node> share(TN, std::vector<Rc<Edge>>);

    public:
        TN const &info() const;
        std::vector<Rc<Edge>> const &inputs() const;
        std::vector<Rc<Edge>> const &outputs() const;
        std::unordered_set<Rc<Node>> predecessors() const;
        std::unordered_set<Rc<Node>> successors() const;
        void connect(count_t, Rc<Edge>);
        void disconnect(count_t);
        void reconnect(Rc<Edge>, Rc<Edge>);
    };

    template<class TN, class TE>
    class LinkedGraph<TN, TE>::Edge : public RefInplace<LinkedGraph<TN, TE>::Edge> {
        friend class LinkedGraph<TN, TE>;

        TE _info;
        Rc<Node> _source;
        std::unordered_map<Rc<Node>, count_t> _targets;

    public:
        explicit Edge(TE);
        TE const &info() const;
        Rc<Node> source() const;
        std::unordered_set<Rc<Node>> targets() const;
    };

#define LINKED_GRAPH_FN template<class TN, class TE> auto LinkedGraph<TN, TE>::
#define LINKED_GRAPH_CONSTRUCTOR template<class TN, class TE> LinkedGraph<TN, TE>::

    LINKED_GRAPH_FN _cleanupNode(Node &node)->void {
        for (auto i : range0_(node._inputs.size())) {
            node.disconnect(i);
        }
        for (auto const &out : node._outputs) {
            out->_source = nullptr;
        }
    }

    LINKED_GRAPH_FN shareEdge(TE info)->Rc<Edge> {
        return Rc<Edge>(new Edge(std::move(info)));
    }

    LINKED_GRAPH_FN toString() const->std::string {
        std::unordered_map<void *, size_t> indices;
        std::stringstream ss;
        auto f = [&indices, &ss](Rc<Edge> const &e) {
            if (e) {
                auto [it, ok] = indices.try_emplace(e.get(), indices.size());
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

    LINKED_GRAPH_FN nodes() const->std::vector<Rc<Node>> const & {
        return _nodes;
    }

    LINKED_GRAPH_FN inputs() const->std::vector<Rc<Edge>> const & {
        return _inputs;
    }

    LINKED_GRAPH_FN outputs() const->std::vector<Rc<Edge>> const & {
        return _outputs;
    }

    LINKED_GRAPH_FN setInputs(std::vector<Rc<Edge>> inputs)->void {
        ASSERT(std::all_of(inputs.begin(), inputs.end(), [](auto const &e) { return e && !e->_source; }),
               "Robbing output from other node is not allowed");
        _inputs = std::move(inputs);
    }

    LINKED_GRAPH_FN setOutputs(std::vector<Rc<Edge>> outputs)->void {
        _outputs = std::move(outputs);
    }

    LINKED_GRAPH_FN replaceInput(Rc<Edge> from, Rc<Edge> to)->void {
        for (auto &i : _inputs) {
            if (i == from) {
                (i = to)->_source = nullptr;
            }
        }
    }

    LINKED_GRAPH_FN replaceOutput(Rc<Edge> from, Rc<Edge> to)->void {
        for (auto &i : _outputs) {
            if (i == from) {
                i = to;
            }
        }
    }

    LINKED_GRAPH_FN pushNode(TN info, std::vector<Rc<Edge>> outputs)->Rc<Node> {
        auto ans = Node::share(std::move(info), std::move(outputs));
        _nodes.push_back(ans);
        return ans;
    }

    LINKED_GRAPH_FN eraseNode(count_t i)->void {
        ASSERT(i < _nodes.size(), "Node index out of range");
        auto it = _nodes.begin() + i;
        _cleanupNode(**it);
        _nodes.erase(it);
    }

    LINKED_GRAPH_FN eraseNode(Rc<Node> node)->void {
        auto it = std::find(_nodes.begin(), _nodes.end(), node);
        if (it == _nodes.end()) { return; }
        _cleanupNode(**it);
        _nodes.erase(it);
    }

    LINKED_GRAPH_FN cleanup(bool useless(TE const &))->size_t {
        std::unordered_set<Edge *> outputs;
        outputs.reserve(_outputs.size());
        std::transform(_outputs.begin(), _outputs.end(), std::inserter(outputs, outputs.end()), [](auto const &e) { return e.get(); });
        auto useful = [&](Rc<Edge> const &e) {
            return !e->_targets.empty() ||     // 还有节点连接到这个边
                   outputs.contains(e.get()) ||// 这个边是全图输出
                   !useless ||                 // 不需要其他判断
                   !useless(e->_info);         // 这个边其他原因有用
        };

        auto before = _nodes.size();
        while (true) {
            auto endit = std::remove_if(
                _nodes.begin(), _nodes.end(),
                [&, this](auto &n) {
                    auto useless_ = std::none_of(n->_outputs.begin(), n->_outputs.end(), useful);
                    if (useless_) { _cleanupNode(*n); }
                    return useless_;
                });
            if (endit == _nodes.end()) {
                break;
            }
            _nodes.erase(endit, _nodes.end());
        }
        return before - _nodes.size();
    }

    LINKED_GRAPH_FN sort()->bool {
        std::vector<Rc<Node>> ans;
        ans.reserve(_nodes.size());
        std::unordered_set<Node *> mapped;
        while (mapped.size() < _nodes.size()) {
            auto before = mapped.size();
            for (auto const &n : _nodes) {
                if (!mapped.contains(n.get()) &&
                    // ∀e ∈ n.inputs, e.source ∈ mapped
                    std::all_of(n->_inputs.begin(), n->_inputs.end(),
                                [&mapped](auto const &e) { return !e || !e->_source || mapped.contains(e->_source.get()); })) {
                    mapped.insert(n.get());
                    ans.push_back(n);// 拷贝 n，因为如果排序失败，不能改变 _nodes
                }
            }
            if (before == mapped.size()) {
                return false;
            }
        }
        _nodes = std::move(ans);
        return true;
    }

    LINKED_GRAPH_FN Node::share(TN info, std::vector<Rc<Edge>> outputs)
        ->Rc<Node> {
        auto ans = Rc<Node>(new Node(std::move(info), std::move(outputs)));
        for (auto &edge : ans->_outputs) {
            ASSERT(!edge->_source, "Robbing output from other node is not allowed")
            edge->_source = ans;
        }
        return ans;
    }

    LINKED_GRAPH_FN Node::info() const->TN const & {
        return _info;
    }

    LINKED_GRAPH_FN Node::inputs() const->std::vector<Rc<Edge>> const & {
        return _inputs;
    }

    LINKED_GRAPH_FN Node::outputs() const->std::vector<Rc<Edge>> const & {
        return _outputs;
    }

    LINKED_GRAPH_FN Node::predecessors() const->std::unordered_set<Rc<Node>> {
        std::unordered_set<Rc<Node>> ans;
        for (auto const &e : _inputs) {
            if (e->_source) {
                ans.insert(e->_source);
            }
        }
        return ans;
    }

    LINKED_GRAPH_FN Node::successors() const->std::unordered_set<Rc<Node>> {
        std::unordered_set<Rc<Node>> ans;
        for (auto const &e : _outputs) {
            for (auto const &[n, _] : e->_targets) {
                ans.insert(n);
            }
        }
        return ans;
    }

    LINKED_GRAPH_FN Node::connect(count_t i, Rc<Edge> input)->void {
        if (i < _inputs.size()) {
            disconnect(i);
        }
        if (i >= _inputs.size()) {
            _inputs.resize(i + 1, nullptr);
        }
        if (input) {
            ++input->_targets.try_emplace(this->shared_from_this(), 0).first->second;
            _inputs.at(i) = std::move(input);
        }
    }

    LINKED_GRAPH_FN Node::disconnect(count_t i)->void {
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

    LINKED_GRAPH_FN Node::reconnect(Rc<Edge> from, Rc<Edge> to)->void {
        for (auto i : range0_(_inputs.size())) {
            if (_inputs[i] == from) {
                connect(i, to);
            }
        }
    }

    LINKED_GRAPH_FN Edge::info() const->TE const & {
        return _info;
    }

    LINKED_GRAPH_FN Edge::source() const->Rc<Node> {
        return _source;
    }

    LINKED_GRAPH_FN Edge::targets() const->std::unordered_set<Rc<Node>> {
        std::unordered_set<Rc<Node>> ans;
        for (auto [n, _] : _targets) {
            ans.emplace(std::move(n));
        }
        return ans;
    }

    LINKED_GRAPH_CONSTRUCTOR Node::Node(TN info, std::vector<Rc<Edge>> outputs)
        : _info(std::move(info)),
          _inputs(),
          _outputs(std::move(outputs)) {}

    LINKED_GRAPH_CONSTRUCTOR Edge::Edge(TE info)
        : _info(std::move(info)),
          _source(),
          _targets() {}

    LINKED_GRAPH_CONSTRUCTOR LinkedGraph(Graph<TN, TE> g)
        : _nodes(g.topology.nodeCount()),
          _inputs(g.topology.globalInputsCount()),
          _outputs() {

        std::vector<Rc<Edge>> edges(g.edges.size());
        std::transform(g.edges.begin(), g.edges.end(), edges.begin(),
                       [](auto &e) { return shareEdge(std::move(e)); });

        auto it = g.topology.begin(), end_ = g.topology.end();
        while (it != end_) {
            auto [nodeIdx, inputs, outputs] = *it++;
            std::vector<Rc<Edge>> outputs_(outputs.size());
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

    LINKED_GRAPH_FN intoGraph() const->Graph<TN, TE> {
        auto topology = GraphTopo(
            static_cast<count_t>(_inputs.size()),
            static_cast<count_t>(_outputs.size()),
            _nodes.size());
        std::vector<TN> nodes;
        std::vector<TE> edges;

        nodes.reserve(_nodes.size());
        edges.reserve(_inputs.size());

        std::unordered_set<Node *> mappedNodes;
        std::unordered_map<void *, GraphTopo::OutputEdge> edgeIndices;
        for (auto &e : _inputs) {
            auto [it, ok] = edgeIndices.try_emplace(e.get(), edgeIndices.size());
            ASSERT(ok, "");
            edges.emplace_back(std::move(e->_info));
        }
        while (mappedNodes.size() < _nodes.size()) {
            auto before = mappedNodes.size();
            for (auto &n : _nodes) {
                if (mappedNodes.contains(n.get())) { continue; }
                // ∃e ∈ n.inputs, e.source ∉ mapped
                if (std::any_of(n->_inputs.begin(), n->_inputs.end(),
                                [&mappedNodes](auto const &e) {
                                    ASSERT(e, "Input edge is not connected");
                                    return e->_source && !mappedNodes.contains(e->_source.get());
                                })) {
                    continue;
                }
                mappedNodes.insert(n.get());
                nodes.emplace_back(std::move(n->_info));

                count_t newLocalCount = 0;
                topology._connections.reserve(topology._connections.size() + n->_inputs.size());
                for (auto &e : n->_inputs) {
                    auto [it, ok] = edgeIndices.try_emplace(e.get(), edgeIndices.size());
                    if (ok) {
                        ASSERT(!e->_source, "Local edge should not have source node");
                        ++newLocalCount;
                        edges.emplace_back(std::move(e->_info));
                    }
                    topology._connections.push_back(it->second);
                }
                for (auto &e : n->_outputs) {
                    auto [it, ok] = edgeIndices.try_emplace(e.get(), edgeIndices.size());
                    ASSERT(ok, "");
                    edges.emplace_back(std::move(e->_info));
                }

                topology._nodes.push_back({
                    newLocalCount,
                    static_cast<count_t>(n->_inputs.size()),
                    static_cast<count_t>(n->_outputs.size()),
                });
            }
            if (before == mappedNodes.size()) {
                RUNTIME_ERROR("Graph is not topo-sortable.");
            }
        }
        std::transform(_outputs.begin(), _outputs.end(),
                       topology._connections.begin(),
                       [&](auto &e) { return edgeIndices.at(e.get()); });
        return {
            std::move(topology),
            std::move(nodes),
            std::move(edges),
        };
    }

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_LINKED_GRAPH_H
