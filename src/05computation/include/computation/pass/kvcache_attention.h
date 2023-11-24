#ifndef COMPUTATION_KVCACHE_ATTENTION_H
#define COMPUTATION_KVCACHE_ATTENTION_H

#include "computation/operators/concat.h"
#include "computation/operators/mat_mul.h"
#include "computation/operators/simple_binary.h"
#include "computation/operators/softmax.h"
#include "computation/operators/transpose.h"
#include "convert.h"
#include "graph.h"
#include "kernel/collectors/simple_binary.h"

namespace refactor::computation {

    /*    concat
            |
          transpose
            |
          matmul
            |
           div
            |
          softmax    concat
              \       /
                matmul
    */
    class KVCacheAttention : public Converter {
        static size_t count = 0;

    public:
        virtual bool execute(std::shared_ptr<GraphMutant> &g) const override {
            for (auto opMatch : g->internal().nodes()) {
                size_t optype = opMatch->info().op->typeId();
                if (optype != MatMul::typeId()) {
                    continue;
                }
                // match the matmul op
                //auto matmulPredecessors = opMatch->predecessors();
                if (opMatch->predecessors().size() != 2) {
                    continue;
                }
                auto matmulInputLeft = opMatch->predecessors()[0];
                auto matmulInputRight = opMatch->predecessors()[1];
                if (matmulInputLeft->info().op->opTypeId() != Softmax::typeId() ||
                    matmulInputRight->info().op->opTypeId() != Concat::typeId()) {
                    continue;
                }
                //auto softmaxPredecessors = matmulInputLeft->predecessors();
                auto concatInputs = matmulInputRight->inputs();
                if (matmulInputLeft->predecessors().size() != 1 || concatInputs.size() != 2) {
                    continue;
                }
                auto softmaxInput = matmulInputLeft->predecessors()[0];
                if (softmaxInput->info().op->opTypeId() != SimpleBinary::typeId(SimpleBinaryType::Div)) {
                    continue;
                }
                if (softmaxInput->predecessors().size() != 1) {
                    continue;
                }
                divInput = softmaxInput->predecessors()[0];
                if (divInput->info().op->opTypeId() != MatMul::typeId()) {
                    continue;
                }
                auto matmulInputs = divInput->inputs();
                if (divInput->predecessors().size() != 2 || matmulInputs.size() != 2) {
                    continue;
                }
                auto matmul1InputLeft = divInput->predecessors()[0];
                auto matmul1InputRight = divInput->predecessors()[1];
                if (matmul1InputLeft->info().op->opTypeId() != SimpleBinary::typeId(SimpleBinaryType::Add) ||
                    matmul1InputRight->info().op->opTypeId() != Transpose::typeId()) {
                    continue;
                }
                if (matmul1InputRight->predecessors().size() != 1) {
                    continue;
                }
                transposeInput = matmul1InputRight->predecessors()[1];
                if (transposeInput->info().op->opTypeId() != Concat::typeId()) {
                    continue;
                }
                auto concatInputs1 = transposeInput->inputs();
                if (concatInputs1.size() != 2) {
                    continue;
                }
                //auto newNode = g->internal().pushNode();
            }
            return true;
        };
    };

}// namespace refactor::computation

#endif// COMPUTATION_KVCACHE_ATTENTION_H