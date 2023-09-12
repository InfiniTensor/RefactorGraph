#include "computation/graph.h"
#include "common/error_handler.h"
#include "computation/tensor.h"
// #include <fmtlog.h>

using namespace refactor::common;
namespace refactor::computation {

    Graph::Graph(graph_topo::Graph<Node, Edge> &&internal)
        : _internal(std::forward<graph_topo::Graph<Node, Edge>>(internal)),
          _variables() {
        collectVariables();
    }

    void Graph::collectVariables() {
        for (auto const &edge : _internal.edges) {
            if (edge) {
                for (auto const &dim : edge->shape) {
                    if (dim.isVariable()) {
                        auto const &var = dim.variable();
                        _variables.try_emplace(var->name, var);
                    }
                }
            }
        }
    }

    std::unordered_set<std::string> Graph::fillEdgeInfo() {
        std::unordered_set<std::string> unknownVariables;// 未知变量，将返回。
        std::unordered_set<size_t> unknownEdges;         // 未知边，有入边未知的节点无法推导。
        // 拓扑遍历
        for (auto [nodeIdx, inputs, outputs] : _internal.topology) {
            // fmt::println("nodes[{}] = {}", nodeIdx, _internal.nodes[nodeIdx]->opType.name());
            // 构造入边
            std::optional<std::vector<Edge>> inputs_(std::in_place);
            {
                inputs_->reserve(inputs.size());
                for (auto i : inputs) {
                    if (unknownEdges.find(i) != unknownEdges.end()) {
                        // 有入边未知，其出边加入未知边，然后跳过节点
                        for (auto j : outputs) { unknownEdges.insert(j); }
                        inputs_ = std::nullopt;
                        break;
                    }
                    inputs_->push_back(_internal.edges[i]);
                }
            }
            if (!inputs_) { continue; }
            // 推导
            auto infered = _internal.nodes[nodeIdx]->infer(std::move(*inputs_));
            if (infered.isErr()) {
                // fmt::println("inference failed");
                // 推导失败，记录未知变量和边
                auto error = infered.unwrapErr();
                if (std::holds_alternative<UnknownVariable>(error.value)) {
                    // fmt::println("unknown variable: {}", std::get<UnknownVariable>(error.value).name);
                    unknownVariables.insert(std::get<UnknownVariable>(error.value).name);
                    for (auto i : outputs) { unknownEdges.insert(i); }
                } else {
                    throw error;
                }
            } else {
                // fmt::println("inference success");
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
