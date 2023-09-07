#include "graph/graph.h"
#include "common/error_handler.h"
#include "graph/edge_info.h"
#include "infer/infer.h"

using namespace refactor::common;

namespace refactor::graph {

    void Graph::fillEdgeInfo() {
        for (auto [nodeIdx, inputs, outputs] : _internal.topology) {
            std::vector<Edge> inputs_(inputs.size());
            std::transform(inputs.begin(), inputs.end(), inputs_.begin(),
                           [this](size_t idx) { return _internal.edges[idx]; });
            auto const &node = _internal.nodes[nodeIdx];
            InferResult infered(Err(InferError("unimplemented")));
            switch (node.operator_().opType.underlying()) {

                default:
                    break;
            }
            if (infered.isErr()) {
                throw infered.unwrapErr();
            } else {
                auto infered_ = infered.unwrap();
                if (infered_.size() < outputs.size()) {
                    OUT_OF_RANGE("outputs more than infered", infered_.size(), outputs.size());
                } else {
                    for (auto i = 0; i < outputs.size(); ++i) {
                        _internal.edges[outputs[i]] = infered_[i];
                    }
                }
            }
        }
    }

}// namespace refactor::graph
