#ifndef GRAPH_H
#define GRAPH_H

#include "edge_info.h"
#include "graph_topo_searcher.hpp"
#include "op_type.h"

namespace refactor::graph {
    struct NodeInfo {
        common::OpType opType;
    };

    class Graph {
        GraphTopoSearcher<NodeInfo, EdgeInfo> topoSearcher;

        void fillEdgeInfo() {
            auto nodes = topoSearcher.nodes();

            for (auto node : nodes) {
                switch (node.info().opType.underlying()) {
                    case common::OpType::Abs:
                        break;

                    default:
                        break;
                }
            }
        }

    public:
    };

}// namespace refactor::graph

#endif// GRAPH_H
