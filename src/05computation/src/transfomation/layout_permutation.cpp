#include "computation/graph.h"
#include "computation/operators/conv.h"
#include "computation/operators/transpose.h"
#include <execution>

namespace refactor::computation {

    std::shared_ptr<Tensor> transposeNHWC(const Tensor &tensor) {
        auto N = tensor.shape[0];
        auto C = tensor.shape[1];
        auto H = tensor.shape[2];
        auto W = tensor.shape[3];
        Shape shape{N, H, W, C};
        auto ans = Tensor::share(tensor.dataType, shape, LayoutType::NHWC);

        if (tensor.data) {
            size_t num = N * C * H * W;
            size_t size = num * tensor.dataType.size();
            auto [data_, dst] = refactor::mem_manager::Blob::share(size);
            const void *src = *(tensor.data);
            std::for_each_n(std::execution::unseq, natural_t(0), num,
                            [&dst, eleSize = tensor.dataType.size(), H, W, C, &src](auto const i) {
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
            ans->data = std::move(data_);
        }
        return ans;
    }

    std::shared_ptr<Tensor> transposeNCHW(const Tensor &tensor) {
        auto N = tensor.shape[0];
        auto C = tensor.shape[3];
        auto H = tensor.shape[1];
        auto W = tensor.shape[2];
        Shape shape{N, C, H, W};
        auto ans = Tensor::share(tensor.dataType, shape, LayoutType::NCHW);
        return ans;
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
                if (!graph.nodes[node_].op) { continue; }
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

        auto count = 0;
        absl::InlinedVector<uint32_t, 4> perm = {0, 2, 3, 1};
        absl::InlinedVector<uint32_t, 4> perm1 = {0, 3, 1, 2};
        auto &g_ = _internal.linked();
        std::unordered_map<void *, refactor::Rc<refactor::graph_topo::LinkedGraph<Node, Edge>::Edge>> tensorMap;
        for (SubgraphId subId = 0; subId < subgraphs_.size(); ++subId) {
            if (subgraphs_[subId].dependent || !subgraphs_[subId].containsConv) {
                // 不存在可以添加transpose算子的子图
                continue;
            }
            for (auto nodeIdx : subgraphs_[subId].nodes) {
                g_.nodes()[nodeIdx]->info().op->transposeTo(LayoutType::NHWC);
                auto inputs_ = g_.nodes()[nodeIdx]->inputs();
                for (auto i : range0_(inputs_.size())) {
                    auto &t = *inputs_[i]->info().tensor;
                    if (inputs_[i]->source()) {
                        if (nodesMap[inputs_[i]->info().name] != subId) {
                            if (auto it = tensorMap.find(inputs_[i].get()); it != tensorMap.end()) {
                                g_.nodes()[nodeIdx]->reconnect(inputs_[i], it->second);
                                continue;
                            }
                            // 不同属于一个子图，前向需添加transpose
                            auto name = fmt::format("layout_transpose{}", count++);
                            auto newNode = g_.pushNode(
                                {std::make_unique<Transpose>(perm), name},
                                {g_.shareEdge({transposeNHWC(t), name + "_out"})});
                            newNode->connect(0, g_.nodes()[nodeIdx]->inputs()[i]);
                            g_.nodes()[nodeIdx]->connect(i, newNode->outputs()[0]);
                            tensorMap.insert({inputs_[i].get(), newNode->outputs()[0]});
                        }
                        continue;
                    }
                    // 没有前驱结点
                    if (t.layout == LayoutType::NCHW) {
                        if (auto it = tensorMap.find(inputs_[i].get()); it != tensorMap.end()) {
                            g_.nodes()[nodeIdx]->reconnect(inputs_[i], it->second);
                            continue;
                        }
                        refactor::Rc<refactor::graph_topo::LinkedGraph<Node, Edge>::Edge> newEdge = nullptr;
                        if (t.data) {
                            //含有数据，需要clone一份新的tensor
                            newEdge = g_.shareEdge({transposeNHWC(t), inputs_[i]->info().name + "_transpose"});
                            g_.nodes()[nodeIdx]->reconnect(inputs_[i], newEdge);
                        } else {
                            // 插入transpose结点
                            auto name = fmt::format("layout_transpose{}", count++);
                            newEdge = g_.shareEdge({transposeNHWC(t), name + "_out"});
                            auto newNode = g_.pushNode(
                                {std::make_unique<Transpose>(perm), name},
                                {newEdge});
                            newNode->connect(0, g_.nodes()[nodeIdx]->inputs()[i]);
                            g_.nodes()[nodeIdx]->connect(i, newNode->outputs()[0]);
                        }
                        tensorMap.insert({inputs_[i].get(), newEdge});
                    }
                }
                auto outputs_ = g_.nodes()[nodeIdx]->outputs();
                for (auto i : range0_(outputs_.size())) {
                    auto &t = *outputs_[i]->info().tensor;
                    if (t.layout == LayoutType::NCHW) {
                        Shape shape = {t.shape[0], t.shape[2], t.shape[3], t.shape[1]};
                        t.layout = LayoutType::NHWC;
                        t.shape = shape;
                    }
                    if (outputs_[i]->targets().size() == 0) {
                        // 当前output为globaloutput
                        auto name = fmt::format("layout_transpose{}", count++);
                        auto newNode = g_.pushNode(
                            {std::make_unique<Transpose>(perm1), name},
                            {g_.shareEdge({transposeNCHW(t), name + "_out"})});
                        newNode->connect(0, g_.nodes()[nodeIdx]->outputs()[i]);
                        g_.replaceOutput(outputs_[i], newNode->outputs()[0]);
                        continue;
                    }
                    for (auto node : outputs_[i]->targets()) {
                        if (nodesMap[node->info().name] != subId) {
                            if (auto it = tensorMap.find(outputs_[i].get()); it != tensorMap.end()) {
                                node->reconnect(outputs_[i], it->second);
                                continue;
                            }
                            // 插入transpose结点
                            auto name = fmt::format("layout_transpose{}", count++);
                            auto newNode = g_.pushNode(
                                {std::make_unique<Transpose>(perm1), name},
                                {g_.shareEdge({transposeNCHW(t), name + "_out"})});
                            newNode->connect(0, g_.nodes()[nodeIdx]->outputs()[i]);
                            auto it = std::find(node->inputs().begin(), node->inputs().end(), outputs_[i]);
                            node->connect(it - node->inputs().begin(), newNode->outputs()[0]);
                            tensorMap.insert({outputs_[i].get(), newNode->outputs()[0]});
                        }
                    }
                }
            }
        }
        fmt::println("Transposed finished");
    }

}// namespace refactor::computation
