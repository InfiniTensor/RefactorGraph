#ifndef GRAPH_TOPO_POLYMORPH_GRAPH_H
#define GRAPH_TOPO_POLYMORPH_GRAPH_H

#include "container.h"
#include "linked_graph.hpp"
#include <variant>

namespace refactor::graph_topo {

    template<class TN, class TE>
    class PolymorphGraph {
        mutable std::variant<Graph<TN, TE>, LinkedGraph<TN, TE>> _internal;

    public:
        PolymorphGraph(Graph<TN, TE> g) noexcept : _internal(std::move(g)) {}
        PolymorphGraph(LinkedGraph<TN, TE> g) noexcept : _internal(std::move(g)) {}

        bool isContiguous() const noexcept { return std::holds_alternative<Graph<TN, TE>>(_internal); }
        bool isLinked() const noexcept { return std::holds_alternative<LinkedGraph<TN, TE>>(_internal); }

        auto contiguous() const noexcept -> Graph<TN, TE> const & {
            if (isLinked()) {
                _internal = std::get<LinkedGraph<TN, TE>>(_internal).intoGraph();
            }
            return std::get<Graph<TN, TE>>(_internal);
        }
        auto linked() const noexcept -> LinkedGraph<TN, TE> const & {
            if (isContiguous()) {
                _internal = LinkedGraph(std::move(std::get<Graph<TN, TE>>(_internal)));
            }
            return std::get<LinkedGraph<TN, TE>>(_internal);
        }

        auto contiguous() noexcept -> Graph<TN, TE> & {
            if (isLinked()) {
                _internal = std::get<LinkedGraph<TN, TE>>(_internal).intoGraph();
            }
            return std::get<Graph<TN, TE>>(_internal);
        }
        auto linked() noexcept -> LinkedGraph<TN, TE> & {
            if (isContiguous()) {
                _internal = LinkedGraph(std::move(std::get<Graph<TN, TE>>(_internal)));
            }
            return std::get<LinkedGraph<TN, TE>>(_internal);
        }
    };

}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_POLYMORPH_GRAPH_H
