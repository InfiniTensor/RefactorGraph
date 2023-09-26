#include "frontend/graph.h"
#include "common/error_handler.h"
#include "frontend/tensor.h"
#include <chrono>
#include <execution>
#include <filesystem>
#include <fmtlog.h>
#include <mutex>

namespace refactor::frontend {
    using namespace common;
    using namespace std::chrono;
    namespace fs = std::filesystem;

    void configLog() {
        static std::once_flag logInitFlag;
        std::call_once(logInitFlag, [] {
            auto dir = fs::path(__FILE__)
                           .parent_path()
                           .parent_path()
                           .parent_path()
                           .parent_path()
                           .append("log");
            fs::create_directory(dir);
            auto name = fmt::format("ver_{}_{}.log", __DATE__, __TIME__);
            for (auto &c : name) {
                if (c == ' ' || c == ':') {
                    c = '_';
                }
            }
            fmtlog::setLogFile(dir.append(name).c_str(), false);
            fmtlog::startPollingThread();
            logi("process start with log file {}", name);
        });
    }

    Graph::Graph(graph_topo::Graph<Node, Edge> internal)
        : _internal(std::move(internal)), _variables() {
        collectVariables();
        configLog();
    }

    bool Graph::substitute(const char *name, int64_t value) {
        if (auto it = _variables.find(name); it != _variables.end()) {
            it->second->value = value;
            return true;
        }
        return false;
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

    std::unordered_set<std::string> Graph::fillEdgeInfo(bool calculate) {
        std::unordered_set<std::string> unknownVariables;// 未知变量，将返回。
        InferOptions options{calculate};
        auto const startTime = high_resolution_clock::now();
        // 拓扑遍历
        for (auto [nodeIdx, inputs, outputs] : _internal.topology) {
            // 构造入边
            bool hasUnknownVariable = false;
            if (std::any_of(inputs.begin(), inputs.end(), [this](auto i) { return !_internal.edges[i].tensor; })) {
                // 有入边未知，跳过节点
                continue;
            }
#ifndef NDEBUG
            logd("infering: {}", _internal.nodes[nodeIdx].name);
#endif
            auto infered = _internal.nodes[nodeIdx].op.infer(TensorRefs(_internal.edges, inputs), options);

            if (infered.isOk()) {
                // 推导成功，填充边信息
                auto infered_ = std::move(infered.unwrap());
                if (infered_.size() < outputs.size()) {
                    OUT_OF_RANGE("outputs more than infered", infered_.size(), outputs.size());
                } else {
#ifndef NDEBUG
                    auto log = fmt::format("node[{}] = {} -> [ ", nodeIdx, _internal.nodes[nodeIdx].name);
                    for (auto const &shape : infered_) {
                        log += shapeFormat(shape->shape);
                        log += ' ';
                    }
                    logd("{}]", log);
#endif

                    std::for_each_n(std::execution::unseq, natural_t(0), outputs.size(),
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
        logi("inference cost time: {} μs", duration_cast<microseconds>(endTime - startTime).count());
#ifndef NDEBUG
        if (unknownVariables.empty()) {
            logGraph();
        }
#endif
        return unknownVariables;
    }

    computation::Graph Graph::lower() const {

        auto const startTime = high_resolution_clock::now();

        std::vector<computation::Edge> edges(_internal.edges.size());
        std::transform(std::execution::unseq,
                       _internal.edges.begin(), _internal.edges.end(), edges.begin(),
                       [](auto const &edge) {
                           using _Tensor = computation::Tensor;
                           computation::Shape shape(edge.tensor->shape.size());
                           std::transform(std::execution::unseq,
                                          edge.tensor->shape.begin(), edge.tensor->shape.end(), shape.begin(),
                                          [](auto const &dim) { return dim.value(); });
                           return computation::Edge{
                               std::make_shared<_Tensor>(_Tensor{edge.tensor->dataType, std::move(shape), edge.tensor->data}),
                               edge.name,
                           };
                       });

        std::vector<computation::Node> nodes(_internal.nodes.size());
        std::transform(_internal.topology.begin(), _internal.topology.end(), nodes.begin(),
                       [this](auto const &nodeRef) {
                           auto const &[op, name] = _internal.nodes[nodeRef.idx];
                           auto op_ = std::all_of(std::execution::unseq,
                                                  nodeRef.outputs.begin(), nodeRef.outputs.end(),
                                                  [this](auto i) { return _internal.edges[i].tensor->hasData(); })
                                          ? nullptr
                                          : op.lower(TensorRefs(_internal.edges, nodeRef.inputs));
                           return computation::Node{std::move(op_), name};
                       });

        auto const endTime = high_resolution_clock::now();
        logi("lowering cost time: {} μs", duration_cast<microseconds>(endTime - startTime).count());

#ifndef NDEBUG
        logi("{}/{} nodes remained after lowering",
             std::count_if(nodes.begin(), nodes.end(), [](auto const &node) { return node.op; }),
             nodes.size());
#endif

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
