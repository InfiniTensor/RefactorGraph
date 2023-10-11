#ifndef GRAPH_TOPO_TEST_TOPO_H
#define GRAPH_TOPO_TEST_TOPO_H

#include "graph_topo/graph_topo.h"

namespace refactor::graph_topo {
    //  `a` (b)
    //    \ /
    // ┌--[A]--┐
    // └(c)-(d)┘(e)
    //   |    \ /
    //   |   ┌[B]┐
    //   |   └(f)┘
    //    \   /
    //     \ /
    //    ┌[C]┐
    //    └`z`┘
    inline auto
    testTopo() -> Builder<const char *, const char *, const char *, const char *> {
        return {
            {
                {"A", {{"a", "b"}, {"c", "d"}}},
                {"B", {{"d", "e"}, {"f"}}},
                {"C", {{"f", "c"}, {"z"}}},
            },    // -> topology
            {"a"},// -> global inputs
            {"z"},// -> global outputs
            {
                {"A", {"*0"}},
                {"B", {"*1"}},
                {"C", {"*2"}},
            },// -> nodes
            {
                {"a", {"|0"}},
                {"b", {"|1"}},
                {"e", {"|4"}},
                {"z", {"!"}},
            },// -> edges
        };
    }
}// namespace refactor::graph_topo

#endif// GRAPH_TOPO_TEST_TOPO_H
