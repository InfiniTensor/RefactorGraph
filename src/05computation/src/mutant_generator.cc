#include "computation/mutant_generator.h"
#include "computation/operators/reshape.h"
#define MAX_SIZE 1024x1024

namespace refactor::computation {
    using K = MutantGenerator;

    void K::init(float equalThreshold_, size_t maxDepth_, OpVec opList_) noexcept {
        equalThreshold = equalThreshold_;
        maxDepth = maxDepth_;
        opList = opList_;
        opFinger.clear();
        opStorage.clear();
        validTensors.clear();
        opHashMaps.clear();
        for (size_t i = 0; i < maxDepth; ++i) {
            opStorage.push_back(opList);
        }
    }

    void K::run(GraphMutant const &inGraph, std::vector<GraphMutant> &outGraphs) noexcept {
        using namespace refactor::graph_topo;

        // init global inputs
        std::unordered_map<size_t, Edge> edges;
        auto edgeIndex = std::vector<size_t>{};
        auto inputs = inGraph.internal().inputs();
        auto outputs = inGraph.internal().outputs();
        ASSERT(outputs.size() == 1, "Do not support more than one output.");
        numValidTensors = inputs.size();
        for (size_t i = 0; i < numValidTensors; ++i) {
            edgeIndex.emplace_back(i);
            edges.insert({i, inputs[i]->info()});
        }
        // init graph
        Builder<size_t, Node, size_t, Edge>
            builder = {{}, edgeIndex, {}, {}, edges};
        GraphMutant curGraph(std::move(builder.build()));
        for (size_t i = 0; i < numValidTensors; ++i) {
            validTensors.emplace_back(curGraph.internal().inputs()[i]);
        }
        dfs(0, inGraph, curGraph, outGraphs);
    }

    void K::dfs(size_t depth, GraphMutant const &inGraph, GraphMutant &curGraph, std::vector<GraphMutant> &outGraphs) noexcept {
        if (is_mutant(curGraph, inGraph)) {
            //存在非全局输出的张量无后继结点，则此图为冗余图
            int count = 0;
            for (size_t i = 0; i < numValidTensors; ++i) {
                if (validTensors[i]->targets().size() == 0) {
                    count++;
                }
            }
            if (count > 1) {
                curGraph.internal().cleanup();
                return;
            }
            auto g = curGraph.clone();
            fmt::println("=======zyz======ok======");
            fmt::println("{}", curGraph.internal().toString([](Node const &o) -> std::string { return std::string(o.op->base->name()); }));
            for (size_t i = 0; i < numValidTensors; ++i) {
                fmt::println("{}. \"{}\" Shape is {}", i, validTensors[i]->info().name,
                             vec2str(validTensors[i]->info().tensor->shape));
            }
            outGraphs.emplace_back(std::move(g));
            curGraph.internal().setOutputs({});
            return;
        }
        if (depth >= maxDepth) {
            return;
        }
        //auto g_ = curGraph.internal();
        for (size_t index = 0; index < opStorage[depth].size(); ++index) {
            auto op = opStorage[depth][index];
            if (op->numInputs == 2) {
                auto opb = dynamic_cast<MyOperator_B *>(op.get());
                for (size_t i = 0; i < numValidTensors; ++i) {
                    for (size_t j = 0; j < numValidTensors; ++j) {
                        if (i == j) {
                            continue;
                        }
                        auto x = validTensors[i]->info().tensor;
                        auto y = validTensors[j]->info().tensor;
                        auto ans = opb->verify(*x, *y);
                        if (ans.size() == 0) {
                            continue;
                        }
                        //fmt::println("{},{}, {}, {}", i, j, reinterpret_cast<void *>(x.get()), reinterpret_cast<void *>(y.get()));
                        auto out = Tensor::share(x->dataType, ans, LayoutType::Others);
                        out->malloc();
                        if (!opb->compute(*x, *y, *out) || have_same_op(op, i, j)) {
                            out->free();
                            continue;
                        }
                        numValidTensors++;
                        opFinger.push_back(op);
                        auto name = fmt::format("{}", depth);
                        auto newEdge = curGraph.internal().shareEdge({out, "tensor_" + name});
                        auto newNode = curGraph.internal().pushNode({op, "op_" + name},
                                                                    {newEdge});
                        newNode->connect(0, validTensors[i]);
                        newNode->connect(1, validTensors[j]);
                        validTensors.push_back(newEdge);
                        // fmt::println("{}", curGraph.internal().toString([](Node const &o) -> std::string { return std::string(o.op->base->name()); }));
                        //fmt::println("{}", reinterpret_cast<void *>(validTensors[j]->info().tensor.get()));
                        dfs(depth + 1, inGraph, curGraph, outGraphs);
                        curGraph.internal().eraseNode(newNode);
                        validTensors.pop_back();
                        opFinger.pop_back();
                        delete_hash_op(op, i, j);
                        numValidTensors--;
                    }
                }
            }
            if (op->numInputs == 1) {
                // if (opFinger.size() != 0 && opFinger.back()->base->opTypeId() == op->base->opTypeId()) {
                //     continue;
                // }
                auto opu = dynamic_cast<MyOperator_U *>(op.get());
                for (size_t i = 0; i < numValidTensors; ++i) {
                    auto x = validTensors[i]->info().tensor;
                    auto ans = opu->verify(*x);
                    if (ans.size() == 0) {
                        continue;
                    }
                    auto out = Tensor::share(x->dataType, ans, LayoutType::Others);
                    out->malloc();
                    if (!opu->compute(*x, *out) || have_same_op(op, i, -1)) {
                        out->free();
                        continue;
                    }
                    numValidTensors++;
                    opFinger.push_back(op);
                    auto name = fmt::format("{}", depth);
                    auto newEdge = curGraph.internal().shareEdge({out, "tensor_" + name});
                    auto newNode = curGraph.internal().pushNode({op, "op_" + name},
                                                                {newEdge});
                    newNode->connect(0, validTensors[i]);
                    validTensors.push_back(newEdge);
                    // fmt::println("{}", curGraph.internal().toString([](Node const &o) -> std::string { return std::string(o.op->base->name()); }));
                    dfs(depth + 1, inGraph, curGraph, outGraphs);
                    curGraph.internal().eraseNode(newNode);
                    validTensors.pop_back();
                    opFinger.pop_back();
                    delete_hash_op(op, i, -1);
                    numValidTensors--;
                }
            }
        }
    }

    bool K::is_mutant(GraphMutant &curGraph, GraphMutant const &inGraph) noexcept {
        if (opFinger.size() != 0 && opFinger.back()->base->opTypeId() == Reshape::typeId()) {
            return false;
        }
        auto inputs = inGraph.internal().inputs();
        auto outputs = inGraph.internal().outputs();
        // fmt::println("=======================output graph =================");
        // fmt::println("{}", curGraph.internal().toString([](Node const &o) -> std::string { return std::string(o.op->base->name()); }));
        // fmt::println("Edges info :");
        // for (size_t i = 0; i < numValidTensors; ++i) {
        //     fmt::println("{}. \"{}\" Shape is {}", i, validTensors[i]->info().name,
        //                  vec2str(validTensors[i]->info().tensor->shape));
        // }
        std::vector<refactor::Rc<refactor::graph_topo::LinkedGraph<Node, Edge>::Edge>> outEdges;
        for (auto output : outputs) {
            int found = -1;
            auto &tensor = *output->info().tensor;
            for (size_t i = inputs.size(); i < validTensors.size(); ++i) {
                if (approx_equal(tensor, *(validTensors[i]->info().tensor))) {
                    found = i;
                    break;
                }
            }
            if (found == -1) {
                // fmt::println("!!!!!!!compare false ");
                return false;
            }
            outEdges.emplace_back(validTensors[found]);
        }
        curGraph.internal().setOutputs(outEdges);
        // fmt::println("=======================compare true =================");
        return true;
    }

    bool K::approx_equal(const Tensor &a, const Tensor &b) const noexcept {
        if (a.shape != b.shape) {
            return false;
        }
        size_t equal = 0, total = 0;
        auto dataA = a.data->get<float>();
        auto dataB = b.data->get<float>();
        for (size_t i = 0; i < a.elementsSize(); ++i) {
            if (dataA[i] == dataB[i]) {
                equal++;
            }
            total++;
        }
        if (float(equal) / total >= equalThreshold) {
            return true;
        }
        return false;
    }

    bool K::have_same_op(Arc<MyOperator> const &op, size_t a, size_t b) noexcept {
        //fmt::println("{}", reinterpret_cast<void *>(op->base.get()));
        std::vector<size_t> hashInfo = {op->base->opTypeId(), a, b};
        auto res = hashVector(hashInfo);
        if (opHashMaps.find(res) != opHashMaps.end()) {
            return true;
        }
        hashInfo = {op->base->opTypeId(), numValidTensors, b};
        auto res1 = hashVector(hashInfo);
        opHashMaps.insert(std::move(res1));
        opHashMaps.insert(std::move(res));
        return false;
    }

    void K::delete_hash_op(Arc<MyOperator> const &op, size_t a, size_t b) noexcept {
        std::vector<size_t> hashInfo = {op->base->opTypeId(), a, b};
        auto res = hashVector(hashInfo);
        if (auto it = opHashMaps.find(res); it != opHashMaps.end()) {
            opHashMaps.erase(it);
        }
        hashInfo = {op->base->opTypeId(), numValidTensors, b};
        auto res1 = hashVector(hashInfo);
        if (auto it = opHashMaps.find(res1); it != opHashMaps.end()) {
            opHashMaps.erase(it);
        }
    }
}// namespace refactor::computation