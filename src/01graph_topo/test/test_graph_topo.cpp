#include "topo.h"
#include <gtest/gtest.h>

using namespace refactor::graph_topo;

TEST(graph_topo, Builder) {
    // fmtlog::setLogLevel(fmtlog::LogLevel::DBG);
    auto [topology, nodes, edges] = testTopo().build();
    auto searcher = Searcher(topology);
    {
        auto const inputs = searcher.globalInputs();
        ASSERT_EQ(inputs.size(), 1);
        EXPECT_EQ(inputs[0].index(), 0);
        EXPECT_EQ(edges[inputs[0].index()], "|0");

        auto const outputs = searcher.globalOutputs();
        ASSERT_EQ(outputs.size(), 1);
        EXPECT_EQ(outputs[0].index(), 6);
        EXPECT_EQ(edges[outputs[0].index()], "!");

        auto const localEdges = searcher.localEdges();
        ASSERT_EQ(localEdges.size(), 2);

        std::unordered_set<size_t> _localEdges;
        std::transform(localEdges.begin(), localEdges.end(), std::inserter(_localEdges, _localEdges.end()),
                       [](auto const &edge) { return edge.index(); });
        ASSERT_EQ(_localEdges.size(), 2);
        EXPECT_EQ(_localEdges, (std::unordered_set<size_t>{1, 4}));

        std::unordered_set<const char *> __localEdges;
        std::transform(localEdges.begin(), localEdges.end(), std::inserter(__localEdges, __localEdges.end()),
                       [&](auto const &edge) { return edges[edge.index()]; });
        ASSERT_EQ(__localEdges.size(), 2);
        EXPECT_NE(__localEdges.find("|1"), __localEdges.end());
        EXPECT_NE(__localEdges.find("|4"), __localEdges.end());
    }
}
