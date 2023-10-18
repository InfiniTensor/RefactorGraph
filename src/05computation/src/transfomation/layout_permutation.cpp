#include "computation/graph.h"
#include "computation/operators/conv.h"
#include "computation/operators/transpose.h"
#include <execution>

namespace refactor::computation {

    void transposeNHWC(std::shared_ptr<Tensor> tensor) {
        int N = tensor->shape[0];
        int C = tensor->shape[1];
        int H = tensor->shape[2];
        int W = tensor->shape[3];
        size_t num = N * C * H * W;
        size_t size = num * tensor->dataType.size();
        auto [data_, dst] = refactor::mem_manager::Blob::share(size);
        const void *src = *(tensor->data);
        std::for_each_n(std::execution::unseq, natural_t(0), num,
                        [&dst, eleSize = tensor->dataType.size(), H, W, C, &src](auto const i) {
                            int newIndex = i;
                            int n = newIndex / (H * W * C);
                            newIndex %= (H * W * C);
                            int h = newIndex / (W * C);
                            newIndex %= (W * C);
                            int w = newIndex / C;
                            int c = newIndex % C;
                            int oldIndex = n * C * H * W + c * H * W + h * W + w;
                            std::memcpy(dst + i * eleSize, src + oldIndex * eleSize, eleSize);
                        });
        tensor->data = data_;
    }

    void Graph::layoutPermute() {
        using SubgraphId = uint_lv1;
        constexpr static auto EXTERNAL = std::numeric_limits<SubgraphId>::max();
        constexpr static auto DEPENDENT = EXTERNAL;
        constexpr static auto INDEPENDENT = DEPENDENT - 1;
        constexpr static auto REAL_SUBGRAPH = INDEPENDENT;

        auto &graph = _internal.contiguous();

        // 用于记录子图。可取 `SubgraphId` 的任何值，但有不同的含义：
        //
        // - DEPENDENT: 布局依赖子图；
        // - INDEPENDENT: 布局无关子图；
        // - else: 用于融合子图。如果发现两个子图相连，则将较大的子图号视作较小的子图号的别名。
        std::vector<SubgraphId> subgraphs;
        // 用于记录每个节点所属的子图号。为了在子图融合中维持拓扑序，不直接记录子图到节点的映射。
        std::vector<SubgraphId> nodes(graph.nodes.size(), EXTERNAL);
        std::unordered_map<std::string, SubgraphId> nodesMap(graph.nodes.size());

        auto searcher = graph_topo::Searcher(graph.topology);
        for (auto node : searcher.nodes()) {
            auto const &[op, name] = graph.nodes[node];
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
            auto const &[op, name] = graph.nodes[nodeIdx];
            if (!op) { continue; }
            auto subgraph = nodes[nodeIdx];
            nodesMap[graph.nodes[nodeIdx].name] = subgraph;
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
                auto const &[op, name] = graph.nodes[nodeIdx];
                msg += fmt::format("{} ", name);
            }
            fmt::println("{}]", msg);
        }

        int count = 0;
        absl::InlinedVector<uint32_t, 4> perm = {0, 2, 3, 1};
        auto &g_ = _internal.linked();
        for (SubgraphId id = 0; id < subgraphs_.size(); ++id) {
            if (subgraphs_[id].dependent || !subgraphs_[id].containsConv) {
                continue;
            }
            for (auto nodeIdx : subgraphs_[id].nodes) {
                g_.nodes()[nodeIdx]->info().op->transposeTo(LayoutType::NHWC);
                auto inputs = g_.nodes()[nodeIdx]->inputs();
                for (size_t i = 0; i < inputs.size(); ++i) {
                    auto e = inputs[i]->info();
                    //同属于一个子图，不需要添加transpose
                    if (inputs[i]->source() == nullptr || nodesMap[inputs[i]->source()->info().name] != id) {
                        if (e.tensor->data && e.tensor->layout == LayoutType::NCHW) {
                            // const fold
                            transposeNHWC(e.tensor);
                            e.tensor->layout = LayoutType::NHWC;
                        } else if (!e.tensor->data) {
                            // insert transpose op
                            Node transpose = {std::make_unique<Transpose>(std::move(perm)), fmt::format("InsertTranspose{}", count)};
                            Shape shape = {e.tensor->shape[0],
                                           e.tensor->shape[2],
                                           e.tensor->shape[3],
                                           e.tensor->shape[1]};
                            Tensor tensor = {e.tensor->dataType, shape, LayoutType::NHWC, nullptr};
                            Edge insertEdge = {std::make_shared<Tensor>(tensor), fmt::format("InsertEdge{}", count++)};
                            auto newNode = g_.pushNode(std::move(transpose), {g_.shareEdge(insertEdge)});
                            newNode->connect(0, g_.nodes()[nodeIdx]->inputs()[i]);
                            g_.nodes()[nodeIdx]->connect(i, newNode->outputs()[0]);
                        } else {
                            continue;
                        }
                    }
                }
                auto outputs = g_.nodes()[nodeIdx]->outputs();
                for (size_t i = 0; i < outputs.size(); ++i) {
                    auto &e = outputs[i]->info();
                    if (e.tensor->layout == LayoutType::NCHW) {
                        e.tensor->layout = LayoutType::NHWC;
                    }
                    if (outputs[i]->targets().size() == 0) {
                        // current edge is global output
                        Node transpose = {std::make_unique<Transpose>(std::move(perm)), fmt::format("InsertTranspose{}", count)};
                        Shape shape = {e.tensor->shape[0],
                                       e.tensor->shape[2],
                                       e.tensor->shape[3],
                                       e.tensor->shape[1]};
                        Tensor tensor = {e.tensor->dataType, shape, LayoutType::NHWC, nullptr};
                        Edge insertEdge = {std::make_shared<Tensor>(tensor), fmt::format("InsertEdge{}", count++)};
                        auto newNode = g_.pushNode(std::move(transpose), {g_.shareEdge(insertEdge)});
                        newNode->connect(0, g_.nodes()[nodeIdx]->outputs()[i]);
                        g_.replaceOutput(outputs[i], newNode->outputs()[0]);
                        continue;
                    }
                    for (auto node : outputs[i]->targets()) {
                        if (nodesMap[node->info().name] != id) {
                            // insert transpose op
                            Node transpose = {std::make_unique<Transpose>(std::move(perm)), fmt::format("InsertTranspose{}", count)};
                            Shape shape = {e.tensor->shape[0],
                                           e.tensor->shape[2],
                                           e.tensor->shape[3],
                                           e.tensor->shape[1]};
                            Tensor tensor = {e.tensor->dataType, shape, LayoutType::NHWC, nullptr};
                            Edge insertEdge = {std::make_shared<Tensor>(tensor), fmt::format("InsertEdge{}", count++)};
                            auto newNode = g_.pushNode(std::move(transpose), {g_.shareEdge(insertEdge)});
                            newNode->connect(0, g_.nodes()[nodeIdx]->outputs()[i]);
                            auto it = std::find(node->inputs().begin(), node->inputs().end(), outputs[i]);
                            node->connect(it - node->inputs().begin(), newNode->outputs()[0]);
                        }
                    }
                }
            }
        }
        fmt::println("Transposed finished");
    }

}// namespace refactor::computation
