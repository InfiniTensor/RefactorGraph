#ifndef FRONTEND_GRAPH_H
#define FRONTEND_GRAPH_H

#include "computation/graph.h"
#include "graph_topo/graph_topo.h"
#include "operator.h"

namespace refactor::frontend {

    class Graph {
        graph_topo::Graph<Node, Edge> _internal;
        std::unordered_map<std::string, DimVariable> _variables;

        void logGraph() const;

    public:
        explicit Graph(graph_topo::Graph<Node, Edge>);
        Graph(Graph const &) = default;
        Graph(Graph &&) = default;

        void collectVariables();
        std::unordered_set<std::string> fillEdgeInfo();
        computation::Graph lower() const;

        auto internal() -> decltype(_internal) &;
        auto internal() const -> decltype(_internal) const &;

        bool substitute(const char *, int64_t);
    };

}// namespace refactor::frontend

#endif// FRONTEND_GRAPH_H
