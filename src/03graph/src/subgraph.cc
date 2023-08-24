#include "graph/graph.h"
#include <map>

using namespace refactor::common;

namespace refactor::graph {
    using GraphTopo_ = GraphTopo<Cell<NodeInfo>, Cell<EdgeInfo>>;
    using OldEdge = GraphTopoSearcher<Cell<NodeInfo>, Cell<EdgeInfo>>::Edge;
    using NewEdge = GraphTopo<Cell<NodeInfo>, Cell<EdgeInfo>>::EdgeRef;

    auto GraphMut::extract(std::vector<std::vector<seacher_t::Node>> const &subgraphs) -> std::vector<seacher_t::Node> {
        std::vector<seacher_t::Node> ans;
        ans.reserve(subgraphs.size());

        for (auto const &subgraph : subgraphs) {
            if (subgraph.empty()) {
                RUNTIME_ERROR("Subgraph cannot be empty");
            }
            if (subgraph.size() > 1) {
                RUNTIME_ERROR("Subgraph with more than 1 node is not supported currently");
            }

            auto const &node = subgraph[0];
            GraphTopo_ newGraphTopo;
            // 对齐子图输入
            auto inputs = node.inputs();
            std::vector<NewEdge> newInputs(inputs.size());
            std::transform(inputs.begin(), inputs.end(), newInputs.begin(),
                           [&newGraphTopo](auto const &input) { return newGraphTopo.addEdge(input.info()); });
            // 创建子图节点
            auto newNode = newGraphTopo.addNode(
                std::move(node.info().value),
                std::move(newInputs),
                std::vector<Cell<EdgeInfo>>(node.outputs().size(), EdgeInfo{}));
            // 对齐子图输出
            std::vector<NewEdge> newOutputs(node.outputs().size());
            std::transform(node.outputs().begin(), node.outputs().end(), newOutputs.begin(),
                           [&newNode](auto const &output) { return newNode[output.index()]; });
            for (auto const &edge : newOutputs) {
                newGraphTopo.markOutput(edge);
            }
            // 创建子图
            auto graph = std::make_shared<GraphMut>(std::move(newGraphTopo));
            graph->fillEdgeInfo();
            {// 验证子图推理
                auto outputs = node.outputs();
                auto newOutputs_ = graph->topo().globalOutputs();
                for (size_t i = 0; i < newOutputs.size(); ++i) {
                    ASSERT(outputs[i].info().value == newOutputs_[i].info().value,
                           "Infered input info mismatch");
                }
            }
            node.info().value = Subgraph{std::move(graph)};
            ans.push_back(node);
        }

        return ans;
    }

    void GraphMut::reduce() {
        GraphTopo_ newTopo;
        std::map<OldEdge, NewEdge> old2new;
        // 克隆全图输入
        for (auto const &input : _topo.globalInputs()) {
            old2new.insert({input, newTopo.addEdge({std::move(input.info().value)})});
        }
        // 克隆节点
        for (auto const &node : _topo.nodes()) {
            if (node.info().value.isOperator()) {// 克隆算子
                // 对齐算子输入
                auto const &inputs = node.inputs();
                std::vector<NewEdge> newInputs(inputs.size());
                std::transform(inputs.begin(), inputs.end(), newInputs.begin(),
                               [&old2new](auto const &input) { return old2new[input]; });
                // 导出算子输出信息
                auto const &outputs = node.outputs();
                std::vector<Cell<EdgeInfo>> newOutputs;
                newOutputs.reserve(outputs.size());
                std::transform(outputs.begin(), outputs.end(), std::back_inserter(newOutputs),
                               [](auto const &output) { return std::move(output.info().value); });
                // 添加算子
                auto newNode = newTopo.addNode(std::move(node.info().value), std::move(newInputs), std::move(newOutputs));
                // 对齐算子输出
                for (auto i = 0; i < outputs.size(); ++i) { old2new.insert({outputs[i], newNode[i]}); }

            } else if (node.info().value.isSubgraph()) {// 展平子图
                auto subgraph = std::move(node.info().value.subgraph().graph);
                std::map<OldEdge, NewEdge> inner2new;

                {// 对齐子图输入
                    auto const &inputs = node.inputs();
                    auto innerInputs = subgraph->topo().globalInputs();
                    for (size_t i = 0; i < inputs.size(); ++i) {
                        inner2new[innerInputs[i]] = old2new[inputs[i]];
                    }
                }
                for (auto const &node_ : subgraph->topo().nodes()) {
                    // 克隆子图算子
                    // 子图算子的输出不会被用在子图之外，所以不需要对齐到新图

                    // 对齐算子输入
                    auto const &inputs_ = node_.inputs();
                    std::vector<NewEdge> newInputs(inputs_.size());
                    std::transform(inputs_.begin(), inputs_.end(), newInputs.begin(),
                                   [&inner2new](auto const &input) { return inner2new[input]; });
                    // 导出算子输出信息
                    auto const &outputs = node_.outputs();
                    std::vector<Cell<EdgeInfo>> newOutputs;
                    newOutputs.reserve(outputs.size());
                    std::transform(outputs.begin(), outputs.end(), std::back_inserter(newOutputs),
                                   [](auto const &output) { return std::move(output.info().value); });
                    // 添加算子
                    auto newNode = newTopo.addNode(std::move(node.info().value), std::move(newInputs), std::move(newOutputs));
                    // 对齐算子输出
                    for (auto i = 0; i < outputs.size(); ++i) { inner2new.insert({outputs[i], newNode[i]}); }
                }
                {// 对齐子图输出
                    auto outputs = node.outputs();
                    auto innerOutputs = subgraph->topo().globalOutputs();
                    for (size_t i = 0; i < outputs.size(); ++i) {
                        old2new[outputs[i]] = inner2new[innerOutputs[i]];
                    }
                }

            } else {
                RUNTIME_ERROR("Unreachable");
            }
        }
        {// 标记全图输出
            auto outputs = _topo.globalOutputs();
            std::vector<NewEdge> newOutputs(outputs.size());
            std::transform(outputs.begin(), outputs.end(), newOutputs.begin(),
                           [&old2new](auto const &output) { return old2new[output]; });
            newTopo.markOutput(newOutputs);
        }
        _topo = GraphTopoSearcher(std::move(newTopo));
    }

}// namespace refactor::graph
