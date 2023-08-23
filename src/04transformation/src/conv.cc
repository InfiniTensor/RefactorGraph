#include "transformation/conv.h"

using namespace refactor::common;
using namespace refactor::graph;

namespace refactor::transformation {

    ConvTransformer::ConvTransformer(graph::GraphMut &&graph)
        : _graph(std::forward<GraphMut>(graph)),
          _convs{} {
        for (auto const &node : _graph.topo().nodes()) {
            auto &info = node.info().value;
            if (info.isSubgraph() || info.operator_().opType != OpType::Conv) {
                continue;
            }

            auto const &kernel = node.inputs()[1].info().value.tensor();
            auto const &dilations = info.operator_().attributes["dilations"].ints();
            if (kernel.shape.size() == 4 && kernel.shape[2] == 1 && kernel.shape[3] == 1) {
                _convs[node] = Rule::_1x1;
            } else if (std::any_of(dilations.begin(), dilations.end(), [](auto x) { return x > 1; })) {
                _convs[node] = Rule::Dilation;
            } else {
                continue;
            }

            GraphTopo<Cell<NodeInfo>, Cell<EdgeInfo>> g;
            std::vector<decltype(g)::EdgeRef> inputs;
            for (auto const &edge : node.inputs()) {
                inputs.push_back(g.addEdge(edge.info()));
            }
            g.markOutput(g.addNode(std::move(info), std::move(inputs), {{}})[0]);
            auto subgraph = std::make_shared<GraphMut>(std::move(g));
            subgraph->fillEdgeInfo();
            info = Subgraph{std::move(subgraph)};
        }
    }

}// namespace refactor::transformation
