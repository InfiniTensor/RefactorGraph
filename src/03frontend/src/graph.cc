﻿#include "frontend/graph.h"
#include "common/error_handler.h"
#include "frontend/tensor.h"
#include <chrono>
#include <execution>
#include <filesystem>
#include <fmtlog.h>
#include <mutex>

using namespace refactor::common;
using namespace std::chrono;
namespace fs = std::filesystem;

namespace refactor::frontend {

    Graph::Graph(graph_topo::Graph<Node, Edge> internal)
        : _internal(std::move(internal)), _variables() {
        collectVariables();

        static std::once_flag logInitFlag;
        std::call_once(logInitFlag, [] {
            auto dir = fs::path(__FILE__)
                           .parent_path()
                           .parent_path()
                           .parent_path()
                           .parent_path()
                           .append("log");
            fs::create_directory(dir);
            fmtlog::setLogFile(dir.append(fmt::format("ver_{}.log", __TIME__)).c_str(), false);
            fmtlog::startPollingThread();
        });
    }

    void Graph::collectVariables() {
        auto const globalInputsCount = _internal.topology.globalInputsCount();
        for (auto i : range0_(globalInputsCount)) {
            auto const &edge = _internal.edges[i];
            if (!edge.tensor) {
                continue;
            }
            std::unordered_set<DimVariable> depVariables;
            for (auto &dim : edge.tensor->shape) {
                if (dim.isVariable()) {
                    auto const &var = dim.variable();
                    if (auto [it, ok] = depVariables.emplace(var); ok) {
                        // varibales with same name is same variable
                        if (auto [it, ok] = _variables.try_emplace(var->name, var); !ok) {
                            dim = DimExpr(it->second);
                        }
                    }
                }
            }
            edge.tensor->depVariables = std::move(depVariables);
        }
        for (auto i : range(globalInputsCount, _internal.edges.size())) {
            auto const &edge = _internal.edges[i];
            if (!edge.tensor) {
                continue;
            }
            // ASSERT(if edge is local, edge has no variable)
        }
    }

    auto Graph::internal() -> decltype(_internal) & { return _internal; }
    auto Graph::internal() const -> decltype(_internal) const & { return _internal; }

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
        logi("----------------------------------");
        auto const startTime = high_resolution_clock::now();
        // 拓扑遍历
        for (auto [nodeIdx, inputs, outputs] : _internal.topology) {
            // 构造入边
            std::optional<std::vector<std::shared_ptr<Tensor>>> inputs_(std::in_place);

            inputs_->reserve(inputs.size());
            for (auto i : inputs) {
                auto input = _internal.edges[i].tensor;
                if (!input) {
                    // 无入边，跳过节点
                    inputs_ = std::nullopt;
                    break;
                }
                inputs_->emplace_back(std::move(input));
            }
            if (!inputs_) { continue; }

            auto infered = _internal.nodes[nodeIdx].op.infer(std::move(*inputs_));

            if (infered.isOk()) {
                // 推导成功，填充边信息
                auto infered_ = std::move(infered.unwrap());
                if (infered_.size() < outputs.size()) {
                    OUT_OF_RANGE("outputs more than infered", infered_.size(), outputs.size());
                } else {
                    std::for_each_n(std::execution::par_unseq, natural_t(0), outputs.size(),
                                    [&infered_, outputs, this](auto i) {
                                        _internal.edges[outputs[i]].tensor = std::move(infered_[i]);
                                    });
                }
            } else {
                // 推导失败，记录未知变量
                auto error = std::move(infered.unwrapErr());
                if (std::holds_alternative<UnknownVariable>(error.value)) {
                    unknownVariables.insert(std::get<UnknownVariable>(error.value).name);
                } else {
                    throw error;
                }
            }
        }
        auto const endTime = high_resolution_clock::now();
        logi("inference cost time: {}μs", duration_cast<microseconds>(endTime - startTime).count());
#ifndef NDEBUG
        if (unknownVariables.empty()) {
            logGraph();
        }
#endif
        return unknownVariables;
    }

    computation::Graph Graph::lower() const {

        std::vector<computation::Edge> edges(_internal.edges.size());
        std::transform(_internal.edges.begin(), _internal.edges.end(), edges.begin(),
                       [](auto const &edge) {
                           using _Tensor = computation::Tensor;
                           computation::Shape shape(edge.tensor->shape.size());
                           std::transform(edge.tensor->shape.begin(), edge.tensor->shape.end(), shape.begin(),
                                          [](auto const &dim) { return dim.value(); });
                           return computation::Edge{
                               std::make_shared<_Tensor>(_Tensor{edge.tensor->dataType, std::move(shape), edge.tensor->data}),
                               edge.name,
                           };
                       });

        std::vector<computation::Node> nodes(_internal.nodes.size());
        std::transform(_internal.topology.begin(), _internal.topology.end(), nodes.begin(),
                       [this](auto const &nodeRef) {
                           auto const &node = _internal.nodes[nodeRef.idx];
                           computation::SharedOp op = nullptr;
                           if (!std::all_of(nodeRef.outputs.begin(), nodeRef.outputs.end(),
                                            [this](auto i) { return _internal.edges[i].tensor->hasData(); })) {
                               std::vector<std::shared_ptr<Tensor>> inputs(nodeRef.inputs.size());
                               std::transform(nodeRef.inputs.begin(), nodeRef.inputs.end(), inputs.begin(),
                                              [this](auto i) { return _internal.edges[i].tensor; });
                               op = node.op.lower(std::move(inputs));
                           }
                           return computation::Node{std::move(op), node.name};
                       });

        return {_internal.topology, std::move(nodes), std::move(edges)};
    }

    void Graph::logGraph() const {
        std::unordered_set<std::string_view> frontNodes, dynamicNodes;
        std::unordered_set<size_t> dataEdges;
        auto it = _internal.topology.begin();
        auto const end = _internal.topology.end();
        {
            logi("compute on device: ");
            auto i = 0;
            while (it != end) {
                auto [nodeIdx, inputs, outputs] = *it++;
                if (!std::all_of(outputs.begin(), outputs.end(),
                                 [this](auto i) { return _internal.edges[i].tensor->hasData(); })) {
                    auto node = _internal.nodes[nodeIdx];
                    logi("{:>8}. {}", i++, node.name);
                    auto opType = node.op.opType.name();
                    dynamicNodes.insert(opType);
                    auto front = true;
                    for (auto i : inputs) {
                        if (_internal.edges[i].tensor->hasData()) {
                            dataEdges.insert(i);
                        } else {
                            front = false;
                        }
                    }
                    if (front) {
                        frontNodes.insert(opType);
                    }
                }
            }
        }
        {
            logi("types:");
            auto i = 0;
            for (auto const &node : dynamicNodes) {
                if (frontNodes.erase(node)) {
                    logi("{:>8}.*{}", i++, node);
                } else {
                    logi("{:>8}. {}", i++, node);
                }
            }
        }
        {
            logi("edges to copy:");
            auto i = 0;
            for (auto edgeIdx : dataEdges) {
                auto const &edge = _internal.edges[edgeIdx];
                std::string depVariables = "[ ";
                for (auto const &var : edge.tensor->depVariables) {
                    depVariables += var->name;
                    depVariables += ' ';
                }
                depVariables += ']';
                logi("{:>8}. {} {} ** {}", i++, edge.name, shapeFormat(edge.tensor->shape), depVariables);
            }
        }
        {
            logi("outputs:");
            auto i = 0;
            for (auto edgeIdx : it.globalOutputs()) {
                auto const &edge = _internal.edges[edgeIdx];
                logi("    outputs[{:>2}] = edge[{:>2}] = {} with {}", i++, edgeIdx, edge.name, shapeFormat(edge.tensor->shape));
            }
        }
    }
}// namespace refactor::frontend