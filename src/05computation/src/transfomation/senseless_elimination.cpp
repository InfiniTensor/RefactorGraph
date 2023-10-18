#include "computation/graph.h"

namespace refactor::computation {

    void Graph::senselessEliminate() {
        auto &graph = _internal.linked();
        using TG = std::decay_t<decltype(graph)>;
        std::vector<TG::NodeRc> senseless;
        for (auto &node : graph.nodes()) {
            if (!node->info().op) {
                senseless.push_back(node);
                continue;
            }
            if (!node->info().op->isIdentity()) {
                continue;
            }
            ASSERT(node->inputs().size() == 1, "");
            ASSERT(node->outputs().size() == 1, "");
            senseless.push_back(node);
            auto const &in = node->inputs()[0];
            auto const &out = node->outputs()[0];
            for (auto &successor : out->targets()) {
                successor->reconnect(out, in);
            }
        }
        for (auto node : senseless) {
            graph.eraseNode(node);
        }
    }

}// namespace refactor::computation
