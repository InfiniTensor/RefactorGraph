﻿#include "computation/graph.h"
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
        for (auto &edge : _internal.edges) {
            if (edge.tensor) {
                for (auto &dim : edge.tensor->shape) {
                    if (dim.isVariable()) {
                        auto const &var = dim.variable();
                        auto [it, ok] = _variables.try_emplace(var->name, var);
                        if (!ok) {// varibales with same name is same variable
                            dim = DimExpr(it->second);
                        }
                    }
                }
            }
        }
    }

    bool Graph::substitute(const char *name, int64_t value) {
        if (auto it = _variables.find(name); it != _variables.end()) {
            it->second->value = value;
            return true;
        } else {
            return false;
        }
    }

    std::unordered_set<std::string> Graph::fillEdgeInfo() {
        std::unordered_set<std::string> unknownVariables;// 未知变量，将返回。
        std::unordered_set<size_t> unknownEdges;         // 未知边，有入边未知的节点无法推导。
        // 拓扑遍历
        for (auto [nodeIdx, inputs, outputs] : _internal.topology) {
            // 构造入边
            std::optional<std::vector<std::shared_ptr<Tensor>>> inputs_(std::in_place);
            {
                inputs_->reserve(inputs.size());
                for (auto i : inputs) {
                    if (unknownEdges.find(i) != unknownEdges.end()) {
                        // 有入边未知，其出边加入未知边，然后跳过节点
                        for (auto j : outputs) { unknownEdges.insert(j); }
                        inputs_ = std::nullopt;
                        break;
                    }
                    auto const &input = _internal.edges[i].tensor;
                    ASSERT(input, "input edge not exist");
                    inputs_->emplace_back(input);
                }
            }
            if (!inputs_) { continue; }
            auto const &node = _internal.nodes[nodeIdx];
            fmt::print("nodes[{}] = {}({})", nodeIdx, node.name, node.op->opType.name());

            // 推导
            auto infered = node.op->infer(std::move(*inputs_));
            if (infered.isErr()) {
                fmt::println(", inference failed");
                // 推导失败，记录未知变量和边
                auto error = infered.unwrapErr();
                if (std::holds_alternative<UnknownVariable>(error.value)) {
                    unknownVariables.insert(std::get<UnknownVariable>(error.value).name);
                    for (auto i : outputs) { unknownEdges.insert(i); }
                } else {
                    throw error;
                }
            } else {
                // 推导成功，填充边信息
                auto infered_ = infered.unwrap();
                if (infered_.size() < outputs.size()) {
                    OUT_OF_RANGE("outputs more than infered", infered_.size(), outputs.size());
                } else {
                    fmt::print(", outputs = ( ");
                    for (auto const &tensor : infered_) {
                        fmt::print("{} ", shapeFormat(tensor->shape));
                    }
                    fmt::println(")");
                    for (auto i = 0; i < outputs.size(); ++i) {
                        _internal.edges[outputs[i]].tensor = std::move(infered_[i]);
                    }
                }
            }
        }
        if (unknownVariables.empty()) {
            std::unordered_set<std::string> dynamicNodes;
            for (auto [nodeIdx, inputs, outputs] : _internal.topology) {
                if (std::any_of(outputs.begin(), outputs.end(), [&](auto i) { return !_internal.edges[i].tensor->hasData(); })) {
                    fmt::println("dynamic node: {}", _internal.nodes[nodeIdx].name);
                    dynamicNodes.insert(std::string(_internal.nodes[nodeIdx].op->opType.name()));
                }
            }
            for (auto const &node : dynamicNodes) {
                fmt::print("{} ", node);
            }
            fmt::println("");
        }
        return unknownVariables;
    }

}// namespace refactor::computation
