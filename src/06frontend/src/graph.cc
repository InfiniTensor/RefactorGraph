#include "frontend/graph.h"
#include "frontend/tensor.h"
#include <chrono>
#include <cstdlib>
#include <execution>
#include <filesystem>
#include <fmtlog.h>
#include <mutex>

namespace refactor::frontend {
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
            if (auto env = std::getenv("LOG_LEVEL"); env) {
                if (std::strcmp(env, "DBG") == 0) {
                    fmtlog::setLogLevel(fmtlog::DBG);
                } else if (std::strcmp(env, "INF") == 0) {
                    fmtlog::setLogLevel(fmtlog::INF);
                } else if (std::strcmp(env, "WRN") == 0) {
                    fmtlog::setLogLevel(fmtlog::WRN);
                } else if (std::strcmp(env, "ERR") == 0) {
                    fmtlog::setLogLevel(fmtlog::ERR);
                } else if (std::strcmp(env, "OFF") == 0) {
                    fmtlog::setLogLevel(fmtlog::OFF);
                }
            }
            fmtlog::setLogFile(dir.append(name).c_str(), false);
            fmtlog::startPollingThread();
            logi("process start with log file {}", name);
        });
    }

    Graph::Graph(graph_topo::Graph<Node, Edge> internal)
        : _internal(std::move(internal)),
          _variables(),
          _edgeSnapshot(_internal.edges.size(), TensorSnapshot{DataType::F32, {}, {}}) {
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
    }
    auto Graph::variables() const -> decltype(_variables) const & {
        return _variables;
    }

    auto Graph::internal() -> decltype(_internal) & { return _internal; }
    auto Graph::internal() const -> decltype(_internal) const & { return _internal; }

    std::unordered_set<std::string> Graph::fillEdgeInfo(bool calculate) {
        std::unordered_set<std::string> unknownVariables;                // 未知变量，将返回。
        std::vector<bool> edgeChanged(_internal.edges.size() * 2, false);// 记录边是否发生变化
        InferOptions options{calculate};
        auto const startTime = high_resolution_clock::now();
        // 拓扑遍历
        for (auto [nodeIdx, inputs, outputs] : _internal.topology) {
            auto unknownEdge = false, inputChanged = false;
            for (auto i : inputs) {
                auto const &input = _internal.edges[i].tensor;
                if (!input) {// 有入边未知
                    unknownEdge = true;
                    break;
                }
                auto checked = edgeChanged[2 * i];    // NOTICE `std::vector<bool>::operator[]` 产生常引用！！！
                auto changed = edgeChanged[2 * i + 1];// NOTICE `std::vector<bool>::operator[]` 产生常引用！！！
                if (!checked) {
                    checked = true;
                    if (changed = _edgeSnapshot[i] != *input) {
                        _edgeSnapshot[i] = input->snapshot();
                    }
                }
                inputChanged |= changed;
            }
            // 有入边未知，跳过节点
            if (unknownEdge) {
                continue;
            }
            if (!inputChanged && std::all_of(outputs.begin(), outputs.end(),
                                             [this](auto i) { return _internal.edges[i].tensor; })) {
                // 入边未发生变化，且出边已推导
                continue;
            }
#ifndef NDEBUG
            logd("infering: {}", _internal.nodes[nodeIdx].name);
#endif
            auto infered = _internal.nodes[nodeIdx].op->infer(TensorRefs(_internal.edges, inputs), options);

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

        std::vector<computation::Node> nodes(_internal.nodes.size());
        std::vector<computation::Edge> edges(_internal.edges.size());
        std::transform(_internal.topology.begin(), _internal.topology.end(), nodes.begin(),
                       [&edges, this](auto const &nodeRef) {
                           auto const &[op, name] = _internal.nodes[nodeRef.idx];
                           auto constant = std::all_of(std::execution::unseq,
                                                       nodeRef.outputs.begin(), nodeRef.outputs.end(),
                                                       [this](auto i) { return _internal.edges[i].tensor->data; });
                           if (constant) {
                               return computation::Node{nullptr, name};
                           }
                           auto fn = [&edges, &nodeRef, this](auto i) {
                               if (edges[i].tensor) {
                                   return;
                               }
                               auto const &[tensor, name] = _internal.edges[i];
                               computation::Shape shape(tensor->shape.size());
                               std::transform(std::execution::unseq,
                                              tensor->shape.begin(), tensor->shape.end(), shape.begin(),
                                              [](auto const &dim) { return dim.value(); });
                               auto layout = shape.size() == 4 ? computation::LayoutType::NCHW : computation::LayoutType::Others;
                               edges[i].tensor = computation::Tensor::share(tensor->dataType, std::move(shape), layout, tensor->data);
                               edges[i].name = name;
                           };
                           auto op_ = op->lower(TensorRefs(_internal.edges, nodeRef.inputs));
                           auto valueDependentInputs = op->valueDependentInputs();
                           auto it = valueDependentInputs.begin();
                           for (auto i : range0_(nodeRef.inputs.size())) {
                               auto input = nodeRef.inputs[i];
                               if (it != valueDependentInputs.end() && i == *it) {
                                   edges[input].name = _internal.edges[input].name;
                                   ++it;
                                   continue;
                               }
                               fn(input);
                           }
                           std::for_each(std::execution::unseq, nodeRef.outputs.begin(), nodeRef.outputs.end(), fn);
                           return computation::Node{std::move(op_), name};
                       });

        auto const endTime = high_resolution_clock::now();
        logi("lowering cost time: {} μs", duration_cast<microseconds>(endTime - startTime).count());

        return {_internal.topology, std::move(nodes), std::move(edges)};
    }

    void Graph::logGraph() const {
        std::unordered_set<std::string_view> frontNodes, dynamicNodes;
        std::unordered_set<size_t> dataEdges;

        logi("compute on device: ");
        for (auto i = 0; auto [nodeIdx, inputs, outputs] : _internal.topology) {
            if (!std::all_of(outputs.begin(), outputs.end(),
                             [this](auto i) { return _internal.edges[i].tensor->data; })) {
                auto const &node = _internal.nodes[nodeIdx];
                logi("{:>8}. {}", i++, node.name);
                auto opType = node.op->opTypeName();
                dynamicNodes.insert(opType);
                auto front = true;
                for (auto i : inputs) {
                    if (_internal.edges[i].tensor->data) {
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

        logi("types:");
        for (auto i = 0; auto const &node : dynamicNodes) {
            if (frontNodes.erase(node)) {
                logi("{:>8}.*{}", i++, node);
            } else {
                logi("{:>8}. {}", i++, node);
            }
        }

        logi("edges to copy:");
        for (auto i = 0; auto edgeIdx : dataEdges) {
            auto const &edge = _internal.edges[edgeIdx];
            std::string depVariables = "[ ";
            for (auto const &var : edge.tensor->depVariables) {
                depVariables += var->name;
                depVariables += ' ';
            }
            depVariables += ']';
            logi("{:>8}. {} {} ** {}", i++, edge.name, shapeFormat(edge.tensor->shape), depVariables);
        }

        logi("outputs:");
        for (auto i = 0; auto edgeIdx : _internal.topology.globalOutputs()) {
            auto const &edge = _internal.edges[edgeIdx];
            logi("    outputs[{:>2}] = edge[{:>2}] = {} with {}", i++, edgeIdx, edge.name, shapeFormat(edge.tensor->shape));
        }
    }
}// namespace refactor::frontend
