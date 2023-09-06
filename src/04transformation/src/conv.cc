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
                } else if (MeetDilationConv(node)) {
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

    /*
 *   input         weight
 *     |             |
 *     |             |
 *   transpose     transpose
 *     |             |
 *     |             |
 *   reshape      reshape  
 *        \       /
 *         \     /
 *          matmul
 *            |
 *          reshape
 *            |
 *          transpose
 *            |
 *          output
 */
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

    /*
 *	input(N,C,H,W)
 *	  |            
 * reshape(N,C,H/d1,d1,W/d2,d2)
 *    |           
 *    |          
 * transpose(N,d1,d2,C,H/d1,W/d2)
 *    |       
 *    |       
 * reshape(N*d1*d2,C,H/d1,W/d2)
 *     \     weight(C,F,k1,k2)
 *      \   /
 *      conv(padding=1,dilation=1)
 *        |
 *     reshape(N,d1,d2,F,H',W')
 *        |
 *     transpose(N,F,H',d1,W',d2)
 *        |
 *     reshape(N,F,H,W)
 *        |
 *      output
 */
    void transformDilationConv(Searcher_::Node &node) {
        auto subgraph = node.info().value.subgraph().graph;
        GraphTopo_ newTopo;
        // get subgraph inputs
        auto globalinputs = subgraph->topoMut().globalInputs();
        auto globaloutputs = subgraph->topoMut().globalOutputs();
        std::vector<GraphTopo_::EdgeRef> edgerefs;
        for (auto i : globalinputs) {
            edgerefs.emplace_back(newTopo.addEdge({std::move(i.info().value)}));
        }
        auto input = newTopo.getInfo(edgerefs[0]);
        auto inputDim = input.value.tensor().shape;
        auto outputDim = globaloutputs[0].info().value.tensor().shape;
        auto dilation = subgraph->topoMut().nodes()[0].info().value.operator_().attributes["dilations"].ints();
        //		auto dilation = node.info().value.operator_().attributes["dilations"].ints();
        auto NullEdge = std::vector<Cell<EdgeInfo>>(1, EdgeInfo{Tensor{}});
        // create conv input->reshape->transpose->reshape
        ShapeVariable reshapeDim1 = {Shape{inputDim[0], inputDim[1], inputDim[2] / dilation[0], dilation[0], inputDim[3] / dilation[1], dilation[1]}};
        auto reshapeAnotherInput1 = newTopo.addEdge({std::move(EdgeInfo(reshapeDim1))});
        Operator reshape = {OpType::Reshape, {}};
        auto reshape1 = newTopo.addNode(NodeInfo(std::move(reshape)), {edgerefs[0], reshapeAnotherInput1}, {NullEdge});
        absl::InlinedVector<long long, 4> perm_vector1 = {0, 3, 5, 1, 2, 4};
        Operator transposePerm1 = {OpType::Transpose, {{"perm", {perm_vector1}}}};
        auto transpose1 = newTopo.addNode(NodeInfo(std::move(transposePerm1)), {reshape1[0]}, {NullEdge});
        ShapeVariable reshapeDim2 = {Shape{inputDim[0] * dilation[0] * dilation[1], inputDim[1], inputDim[2] / dilation[0], inputDim[3] / dilation[1]}};
        auto reshapeAnotherInput2 = newTopo.addEdge({std::move(EdgeInfo(reshapeDim2))});
        auto reshape2 = newTopo.addNode(NodeInfo(std::move(reshape)), {transpose1[0], reshapeAnotherInput2}, {NullEdge});
        // create conv op
        auto stride = subgraph->topoMut().nodes()[0].info().value.operator_().attributes["strides"].ints();
        absl::InlinedVector<long long, 4> pads = {1, 1, 1, 1};
        absl::InlinedVector<long long, 4> dilations = {1, 1};
        Operator conv = {OpType::Conv, {{"pads", {pads}}, {"dilations", {dilations}}, {"strides", {stride}}}};
        auto conv1 = newTopo.addNode(NodeInfo(std::move(conv)), {reshape2[0], edgerefs[1]}, {NullEdge});
        // create conv output->reshape->transpose->reshape
        ShapeVariable reshapeDim3 = {Shape{inputDim[0], dilation[0], dilation[1], outputDim[1], inputDim[2] / dilation[0], inputDim[3] / dilation[1]}};
        auto reshapeAnotherInput3 = newTopo.addEdge({std::move(EdgeInfo(reshapeDim3))});
        auto reshape3 = newTopo.addNode(NodeInfo(std::move(reshape)), {conv1[0], reshapeAnotherInput3}, {NullEdge});
        absl::InlinedVector<long long, 4> perm_vector2 = {0, 3, 4, 1, 5, 2};
        Operator transposePerm2 = {OpType::Transpose, {{"perm", {perm_vector2}}}};
        auto transpose2 = newTopo.addNode(NodeInfo(std::move(transposePerm2)), {reshape3[0]}, {NullEdge});
        ShapeVariable reshapeDim4 = {Shape{outputDim[0], outputDim[1], outputDim[2], outputDim[3]}};
        auto reshapeAnotherInput4 = newTopo.addEdge({std::move(EdgeInfo(reshapeDim4))});
        auto reshape4 = newTopo.addNode(NodeInfo(std::move(reshape)), {transpose2[0], reshapeAnotherInput4}, {NullEdge});
        newTopo.markOutput({reshape4[0]});
        auto newGraphMut = std::make_shared<GraphMut>(std::move(newTopo));
        newGraphMut->fillEdgeInfo();
        node.info().value.subgraph().graph = newGraphMut;
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

    bool MeetDilationConv(Searcher_::Node node) {
        bool flag = false;
        auto &info = node.info().value;
        auto const &dilations = info.operator_().attributes["dilations"].ints();
        auto const &pads = info.operator_().attributes["pads"].ints();
        auto const &strides = info.operator_().attributes["strides"].ints();
        auto const &kernels = node.inputs()[1].info().value.tensor().shape;
        auto const &input = node.inputs()[0].info().value.tensor().shape;
        if (kernels[2] != 3 || kernels[3] != 3) {
            return false;
        }
        if (input[2] % dilations[0] != 0 || input[3] % dilations[1] != 0) {
            return false;
        }
        if (std::any_of(dilations.begin(), dilations.end(), [](auto x) { return x > 1; }) &&
            std::any_of(strides.begin(), strides.end(), [](auto x) { return x == 1; }) &&
            pads[0] == pads[2] && pads[1] == pads[3] && pads[0] == dilations[0] && pads[1] == dilations[1]) {
            flag = true;
        }
        return flag;
    }
}// namespace refactor::transformation
