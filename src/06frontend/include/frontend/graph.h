#ifndef FRONTEND_GRAPH_H
#define FRONTEND_GRAPH_H

#include "computation/graph.h"
#include "graph_topo.h"
#include "operator.h"

namespace refactor::frontend {

    void configLog();

    class Graph {
        graph_topo::Graph<Node, Edge> _internal;
        std::unordered_map<std::string, DimVariable> _variables;
        std::vector<TensorSnapshot> _edgeSnapshot;

        void logGraph() const;

    public:
        explicit Graph(graph_topo::Graph<Node, Edge>);
        Graph(Graph const &) = default;
        Graph(Graph &&) = default;

        bool substitute(const char *, int64_t);
        void collectVariables();
        decltype(_variables) const &variables() const;
        std::unordered_set<std::string> fillEdgeInfo(bool);

        computation::Graph lower() const;

        auto internal() -> decltype(_internal) &;
        auto internal() const -> decltype(_internal) const &;
    };

}// namespace refactor::frontend

#endif// FRONTEND_GRAPH_H
