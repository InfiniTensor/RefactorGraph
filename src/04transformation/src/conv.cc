#include "transformation/conv.h"

using namespace refactor::common;
using namespace refactor::graph;

namespace refactor::transformation {
    using GraphTopo_ = GraphTopo<Cell<NodeInfo>, Cell<EdgeInfo>>;
    using Searcher_ = GraphTopoSearcher<Cell<NodeInfo>, Cell<EdgeInfo>>;

    ConvTransformer::ConvTransformer(graph::GraphMut &&graph)
        : _graph(std::forward<GraphMut>(graph)),
          _convs{} {
        for (auto const &node : _graph.topoMut().nodes()) {
            auto &info = node.info().value;
            if (!info.isSubgraph() && info.operator_().opType == OpType::Conv) {
                if (auto const &kernel = node.inputs()[1].info().value.tensor();
                    kernel.shape.size() == 4 && kernel.shape[2] == 1 && kernel.shape[3] == 1) {
                    _convs[node] = Rule::_1x1;
                } else if (auto const &dilations = info.operator_().attributes["dilations"].ints();
                           std::any_of(dilations.begin(), dilations.end(), [](auto x) { return x > 1; })) {
                    _convs[node] = Rule::Dilation;
                }
            }
        }
        std::vector<std::vector<Searcher_::Node>> subgraphs;
        for (auto const &[node, _] : _convs) {
            subgraphs.push_back({node});
        }
        _graph.extract(std::move(subgraphs));
    }

    void transformConv11(Searcher_::Node &node) {
        auto subgraph = node.info().value.subgraph().graph;
        GraphTopo_ newTopo;
        // get subgraph inputs
        auto globalinputs = subgraph->topoMut().globalInputs();
        std::vector<GraphTopo_::EdgeRef> edgerefs;
        for (auto i : globalinputs) {
            edgerefs.emplace_back(newTopo.addEdge({std::move(i.info().value)}));
        }
        auto input = newTopo.getInfo(edgerefs[0]);
        auto inputDim = input.value.tensor().shape;
        auto weight = newTopo.getInfo(edgerefs[1]);
        auto weightDim = weight.value.tensor().shape;
        auto NullEdge = std::vector<Cell<EdgeInfo>>(1, EdgeInfo{Tensor{}});
        // create transpose op
        absl::InlinedVector<long long, 4> perm_vector1 = {0, 2, 3, 1};
        Operator transposeInput = {OpType::Transpose, {{"perm", {perm_vector1}}}};
        auto transpose1 = newTopo.addNode(NodeInfo(std::move(transposeInput)), {edgerefs[0]}, {NullEdge});
        absl::InlinedVector<long long, 4> perm_vector2 = {1, 0, 2, 3};
        Operator transposeWeight = {OpType::Transpose, {{"perm", {perm_vector2}}}};
        auto transpose2 = newTopo.addNode(NodeInfo(std::move(transposeWeight)), {edgerefs[1]}, {NullEdge});
        // create reshape op
        ShapeVariable reshapeDim1 = {Shape{inputDim[0] * inputDim[2] * inputDim[3], inputDim[1]}};
        ShapeVariable reshapeDim2 = {Shape{weightDim[1], weightDim[0] * weightDim[2] * weightDim[3]}};
        auto reshapeAnotherInput1 = newTopo.addEdge({std::move(EdgeInfo(reshapeDim1))});
        auto reshapeAnotherInput2 = newTopo.addEdge({std::move(EdgeInfo(reshapeDim2))});
        Operator reshape = {OpType::Reshape, {}};
        auto reshape1 = newTopo.addNode(NodeInfo(std::move(reshape)), {transpose1[0], reshapeAnotherInput1}, {NullEdge});
        auto reshape2 = newTopo.addNode(NodeInfo(std::move(reshape)), {transpose2[0], reshapeAnotherInput2}, {NullEdge});
        // create Gemm
        Operator gemm_ = {OpType::Gemm, {}};
        auto gemm = newTopo.addNode(NodeInfo(std::move(gemm_)), {reshape1[0], reshape2[0]}, {NullEdge});
        // create reshape
        ShapeVariable reshapeDim3 = {Shape{inputDim[0], inputDim[2], inputDim[3], weightDim[0]}};
        auto reshapeAnotherInput3 = newTopo.addEdge({std::move(EdgeInfo(reshapeDim3))});
        auto reshape3 = newTopo.addNode(NodeInfo(std::move(reshape)), {gemm[0], reshapeAnotherInput3}, {NullEdge});
        // create transpose
        absl::InlinedVector<long long, 4> perm_vector3 = {0, 3, 1, 2};
        Operator transposeOutput = {OpType::Transpose, {{"perm", {perm_vector3}}}};
        auto transpose3 = newTopo.addNode(NodeInfo(std::move(transposeOutput)), {reshape3[0]}, {NullEdge});
        newTopo.markOutput({transpose3[0]});
        auto newGraphMut = std::make_shared<GraphMut>(std::move(newTopo));
        newGraphMut->fillEdgeInfo();
        node.info().value.subgraph().graph = newGraphMut;
        return;
    }

    void transformDilationConv(Searcher_::Node &node) {
        return;
    }

    void ConvTransformer::transform() {
        // optimize 1x1 conv and dilation conv
        std::map<Searcher_::Node, Rule>::iterator it = _convs.begin();
        for (auto node : _graph.topoMut().nodes()) {
            if (node.index() == it->first.index()) {
                auto info = node.info().value;
                ASSERT(info.isSubgraph(), "Current Node Not A Subgraph Node");
                switch (it->second) {
                    case Rule::_1x1:
                        transformConv11(node);
                        break;
                    case Rule::Dilation:
                        transformDilationConv(node);
                        break;
                    default:
                        TODO("Don't support conv transform type");
                }
                ++it;
            }
        }
    }

}// namespace refactor::transformation
