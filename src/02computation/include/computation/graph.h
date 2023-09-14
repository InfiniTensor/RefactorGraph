﻿#ifndef COMPUTATION_GRAPH_H
#define COMPUTATION_GRAPH_H

#include "graph_topo/graph_topo.h"
#include "operator.h"

namespace refactor::computation {

    class Graph {
        graph_topo::Graph<Node, Edge> _internal;
        std::unordered_map<std::string, DimVariable> _variables;

    public:
        explicit Graph(graph_topo::Graph<Node, Edge> &&);
        void collectVariables();
        bool substitute(const char *, int64_t);
        std::unordered_set<std::string> fillEdgeInfo();
    };

}// namespace refactor::computation

#endif// COMPUTATION_GRAPH_H