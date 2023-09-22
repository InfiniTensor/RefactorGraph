#include "frontend/graph.h"
#include "common/error_handler.h"
#include "frontend/tensor.h"
#include <chrono>
#include <fmtlog.h>

using namespace refactor::common;
using namespace std::chrono;

namespace refactor::frontend {

    Graph::Graph(graph_topo::Graph<Node, Edge> internal)
        : _internal(std::move(internal)), _variables() {
        collectVariables();
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

    class LogGuard {
    public:
        ~LogGuard() {
            fmtlog::poll();
        }
    };

    std::unordered_set<std::string> Graph::fillEdgeInfo() {
        std::unordered_set<std::string> unknownVariables;// 未知变量，将返回。
        LogGuard _logGuard;
        logi("edge inference start");
        auto const startTime = steady_clock::now();
        // 拓扑遍历
        for (auto [nodeIdx, inputs, outputs] : _internal.topology) {
            // 构造入边
            std::optional<std::vector<std::shared_ptr<Tensor>>> inputs_(std::in_place);

            inputs_->reserve(inputs.size());
            for (auto i : inputs) {
                if (!_internal.edges[i].tensor) {
                    // 无入边，跳过节点
                    inputs_ = std::nullopt;
                    break;
                }
                auto const &input = _internal.edges[i].tensor;
                ASSERT(input, "input edge not exist");
                inputs_->emplace_back(input);
            }
            if (!inputs_) { continue; }

            auto const &node = _internal.nodes[nodeIdx];
            auto msg = fmt::format("nodes[{}] = {}({})", nodeIdx, node.name, node.op.opType.name());
            // 推导
            auto infered = node.op.infer(std::move(*inputs_));
            if (infered.isErr()) {
                msg += ", inference failed";
                // 推导失败，记录未知变量
                auto error = infered.unwrapErr();
                if (std::holds_alternative<UnknownVariable>(error.value)) {
                    unknownVariables.insert(std::get<UnknownVariable>(error.value).name);
                } else {
                    throw error;
                }
            } else {
                // 推导成功，填充边信息
                auto infered_ = infered.unwrap();
                if (infered_.size() < outputs.size()) {
                    OUT_OF_RANGE("outputs more than infered", infered_.size(), outputs.size());
                } else {
                    msg += ", outputs = [ ";
                    for (auto const &tensor : infered_) {
                        msg += shapeFormat(tensor->shape);
                        msg += ' ';
                    }
                    msg += ']';
                    for (auto i : range0_(outputs.size())) {
                        _internal.edges[outputs[i]].tensor = std::move(infered_[i]);
                    }
                }
            }
            logi("{}", std::move(msg));
        }
        auto const endTime = steady_clock::now();
        logi("inference cost time: {}μs", duration_cast<microseconds>(endTime - startTime).count());
        if (unknownVariables.empty()) {
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
                               std::make_shared<_Tensor>(_Tensor{edge.tensor->dataType, std::move(shape)}),
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

}// namespace refactor::frontend
