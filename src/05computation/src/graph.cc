#include "computation/graph.h"
#include "computation/operators/conv.h"
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

    void Graph::transpose() {
        using SubgraphId = uint_lv1;
        constexpr static auto EXTERNAL = std::numeric_limits<SubgraphId>::max();
        constexpr static auto DEPENDENT = EXTERNAL;
        constexpr static auto INDEPENDENT = DEPENDENT - 1;
        constexpr static auto REAL_SUBGRAPH = INDEPENDENT;

        // 用于记录子图。可取 `SubgraphId` 的任何值，但有不同的含义：
        //
        // - DEPENDENT: 布局依赖子图；
        // - INDEPENDENT: 布局无关子图；
        // - else: 用于融合子图。如果发现两个子图相连，则将较大的子图号视作较小的子图号的别名。
        std::vector<SubgraphId> subgraphs;
        // 用于记录每个节点所属的子图号。为了在子图融合中维持拓扑序，不直接记录子图到节点的映射。
        std::vector<SubgraphId> nodes(_internal.nodes.size(), EXTERNAL);

        auto searcher = graph_topo::Searcher(_internal.topology);
        for (auto node : searcher.nodes()) {
            auto const &[op, name] = _internal.nodes[node];
            if (!op) { continue; }
            auto &id = nodes[node];
            auto const type = op->isLayoutDependent() ? DEPENDENT : INDEPENDENT;
            // 遍历前驱节点
            for (auto node_ : node.predecessors()) {
                // 搜索前驱的真子图号
                auto preId = nodes[node_];
                {
                    auto t = subgraphs[preId];
                    while (t < REAL_SUBGRAPH) { t = subgraphs[preId = t]; }
                    if (type != t) { continue; }// 节点与前驱不联通
                }
                // update subgraph id ------------|cur|pre|
                if (id == EXTERNAL) {          // |---|---|
                    id = preId;                // | ? | √ |
                } else if (id < preId) {       // |---|---|
                    subgraphs[preId] = id;     // | √ | × |
                } else if (preId < id) {       // |---|---|
                    id = subgraphs[id] = preId;// | × | √ |
                }                              // |---|---|
            }
            // 没有前驱与当前节点联通
            if (id == EXTERNAL) {
                // 新建子图
                id = subgraphs.size();
                subgraphs.push_back(type);
            }
        }

        struct Subgraph {
            bool dependent;
            bool containsConv;
            std::vector<size_t> nodes;
        };
        std::unordered_map<SubgraphId, SubgraphId> subgraphMap;
        std::vector<Subgraph> subgraphs_;
        for (auto nodeIdx : range0_(nodes.size())) {
            auto const &[op, name] = _internal.nodes[nodeIdx];
            if (!op) { continue; }
            auto subgraph = nodes[nodeIdx];
            auto type = subgraphs[subgraph];
            while (type < REAL_SUBGRAPH) {
                type = subgraphs[subgraph = type];
            }
            if (auto [it, ok] = subgraphMap.try_emplace(subgraph, subgraphs_.size()); ok) {
                subgraphs_.push_back({
                    type == DEPENDENT,
                    op->is<Conv>(),
                    {nodeIdx},
                });
            } else {
                auto &subgraph = subgraphs_[it->second];
                subgraph.containsConv |= op->is<Conv>();
                subgraph.nodes.push_back(nodeIdx);
            }
        }
        for (size_t i = 0; i < subgraphs_.size(); ++i) {
            auto msg = fmt::format("Subgraph {} ({})", i, subgraphs_[i].dependent ? "dependent" : "independent");
            if (subgraphs_[i].containsConv) {
                msg += " (contains conv)";
            }
            msg += " : [ ";
            for (auto nodeIdx : subgraphs_[i].nodes) {
                auto const &[op, name] = _internal.nodes[nodeIdx];
                msg += fmt::format("{} ", name);
            }
            fmt::println("{}]", msg);
        }

        for (auto const &subgraph : subgraphs_) {
            if (subgraph.dependent || !subgraph.containsConv) { continue; }
            for (auto nodeIdx : subgraph.nodes) {
                _internal.nodes[nodeIdx].op->transposeTo(LayoutType::NHWC);
                for (auto edge : searcher.nodes()[nodeIdx].outputs()) {
                    auto &e = _internal.edges[edge];
                    if (e.tensor->layout == LayoutType::NCHW) {
                        e.tensor->layout = LayoutType::NHWC;
                    }
                }
            }
        }
        fmt::println("Transpose finished");
    }

    kernel::Graph Graph::lower(kernel::Target target) const {
        std::vector<kernel::Node> nodes(_internal.nodes.size());
        std::vector<kernel::Edge> edges(_internal.edges.size());

        for (auto [nodeIdx, inputs, outputs] : _internal.topology) {
            auto const &[op, name] = _internal.nodes[nodeIdx];
            if (!op) { continue; }
            kernel::TensorRefs inputs_, outputs_;
            inputs_.reserve(inputs.size());
            outputs_.reserve(outputs.size());
            std::transform(inputs.begin(), inputs.end(),
                           std::back_inserter(inputs_), [&](auto i) {
                               return std::cref(*_internal.edges[i].tensor);
                           });
            std::transform(outputs.begin(), outputs.end(),
                           std::back_inserter(outputs_), [&](auto i) {
                               return std::cref(*_internal.edges[i].tensor);
                           });
            auto candidates = op->candidateKernels(target)->filter(std::move(inputs_), std::move(outputs_));
            ASSERT(!candidates.empty(), "No kernel selected");
            nodes[nodeIdx] = {std::move(candidates.front()), name};
        }

        for (auto i : range0_(edges.size())) {
            auto const &[tensor, name] = _internal.edges[i];
            if (!tensor || !tensor->data) { continue; }
            auto fn = target.memFunc();
            auto blob = mem_manager::ForeignBlob::share(fn, tensor->bytesSize());
            fn.copyHd(*blob, *(tensor->data), tensor->bytesSize());
            edges[i] = {std::move(blob), tensor->bytesSize(), name};
        }

        return kernel::Graph(target, _internal.topology, std::move(nodes), std::move(edges));
    }

}// namespace refactor::computation
