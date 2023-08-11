#ifndef GRAPH_H
#define GRAPH_H

#include "data_type.h"
#include "graph_topo_searcher.hpp"
#include "op_type.h"

namespace refactor::graph {

    struct LayoutDim {
        const char *name;
        len_t size, stride;
    };

    struct NodeInfo {
        common::OpType opType;
    };

    struct EdgeInfo {
        common::DataType dataType;
        std::vector<LayoutDim> layout;
    };

    class Graph {
        GraphTopo<NodeInfo, EdgeInfo> topo;
    };

}// namespace refactor::graph

#endif// GRAPH_H
