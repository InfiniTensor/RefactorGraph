#include "computation/graph.h"
#include "common/error_handler.h"
#include "computation/tensor.h"
#include "infer/infer.h"

using namespace refactor::common;
namespace refactor::computation {

    Graph::Graph(graph_topo::Graph<Node, Edge> &&internal)
        : _internal(std::forward<graph_topo::Graph<Node, Edge>>(internal)) {}

    std::unordered_set<std::string> Graph::fillEdgeInfo() {
        std::unordered_set<std::string> unknownVariables;// 未知变量，将返回。
        std::unordered_set<void *> unknownEdges;         // 未知边，有入边未知的节点无法推导。
        // 拓扑遍历
        for (auto [nodeIdx, inputs, outputs] : _internal.topology) {
            // 构造入边
            std::vector<Edge> inputs_(inputs.size());
            for (auto i = 0; i < inputs.size(); ++i) {
                auto ptr = (inputs_[i] = _internal.edges[inputs[i]]).get();
                if (unknownEdges.find(ptr) != unknownEdges.end()) {
                    // 有入边未知，其出边加入未知边，然后跳过节点
                    std::transform(outputs.begin(), outputs.end(), std::inserter(unknownEdges, unknownEdges.end()),
                                   [this](auto output) { return _internal.edges[output].get(); });
                    continue;
                }
            }
            // 推导
            auto infered = _internal.nodes[nodeIdx]->infer(std::move(inputs_));
            if (infered.isErr()) {
                // 推导失败，记录未知变量和边
                auto error = infered.unwrapErr();
                if (std::holds_alternative<UnknownVariable>(error.value)) {
                    unknownVariables.insert(std::get<UnknownVariable>(error.value).name);
                    std::transform(outputs.begin(), outputs.end(), std::inserter(unknownEdges, unknownEdges.end()),
                                   [this](auto output) { return _internal.edges[output].get(); });
                } else {
                    throw error;
                }
            } else {
                // 推导成功，填充边信息
                auto infered_ = infered.unwrap();
                if (infered_.size() < outputs.size()) {
                    OUT_OF_RANGE("outputs more than infered", infered_.size(), outputs.size());
                } else {
                    for (auto i = 0; i < outputs.size(); ++i) {
                        _internal.edges[outputs[i]] = infered_[i];
                    }
                }
            }
        }
        return unknownVariables;
    }

}// namespace refactor::computation
