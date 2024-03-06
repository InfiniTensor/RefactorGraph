#ifndef COMPUTATION_MATMUL_TRANSPOSE_H
#define COMPUTATION_MATMUL_TRANSPOSE_H

#include "../graph.h"
#include "computation/operators/mat_mul.h"
#include "computation/operators/transpose.h"
#include "computation/pass/converter.h"

namespace refactor::computation {
    class MatMulTransposeFuse : public Converter {
    public:
        virtual bool execute(const std::shared_ptr<GraphMutant> &g) const override {
            auto nodesList = g->internal().nodes();
            for (auto opMatch : nodesList) {
                if (opMatch->info().op == nullptr) {
                    continue;
                }
                size_t optype = opMatch->info().op->opTypeId();
                if (optype != MatMul::typeId()) {
                    continue;
                }
                auto matmulOp = dynamic_cast<MatMul *>(opMatch->info().op.get());
                if (opMatch->predecessors().size() != 0) {
                    for (size_t i = 0; i < opMatch->inputs().size(); ++i) {
                        if (auto preOp = opMatch->inputs()[i]->source();
                            preOp != nullptr && preOp->info().op->opTypeId() == Transpose::typeId()) {
                            auto transposeOp = dynamic_cast<Transpose *>(preOp->info().op.get());
                            auto axis = transposeOp->perm;
                            bool flag = false;
                            if (axis[axis.size() - 1] == axis.size() - 2 && axis[axis.size() - 2] == axis.size() - 1) {
                                flag = true;
                            }
                            for (size_t index = 0; index < axis.size() - 2; ++index) {
                                if (index == axis[index]) {
                                    continue;
                                }
                                flag = false;
                                break;
                            }
                            if (flag) {
                                if (i == 0) {
                                    matmulOp->transA = !matmulOp->transA;
                                } else {
                                    matmulOp->transB = !matmulOp->transB;
                                }
                                opMatch->reconnect(opMatch->inputs()[i], preOp->inputs()[0]);
                                g->internal().eraseNode(preOp);
                            }
                        }
                    }
                }
                if (opMatch->successors().size() == 1) {
                    if (auto postOp = *(opMatch->outputs()[0]->targets().begin());
                        postOp != nullptr && postOp->info().op->opTypeId() == Transpose::typeId()) {
                        auto transposeOp = dynamic_cast<Transpose *>(postOp->info().op.get());
                        auto axis = transposeOp->perm;
                        bool flag = false;
                        if (axis[axis.size() - 1] == axis.size() - 2 && axis[axis.size() - 2] == axis.size() - 1) {
                            flag = true;
                        }
                        for (size_t index = 0; index < axis.size() - 2; ++index) {
                            if (index == axis[index]) {
                                continue;
                            }
                            flag = false;
                            break;
                        }
                        if (flag) {
                            matmulOp->transA = !matmulOp->transA;
                            matmulOp->transB = !matmulOp->transB;
                            auto inputsA = opMatch->inputs()[0];
                            auto inputsB = opMatch->inputs()[1];
                            opMatch->connect(0, inputsB);
                            opMatch->connect(1, inputsA);
                            opMatch->outputs()[0]->info().tensor->shape = postOp->outputs()[0]->info().tensor->shape;
                            if (postOp->outputs()[0]->targets().size() == 0) {// global output
                                g->internal().replaceOutput(postOp->outputs()[0], opMatch->outputs()[0]);
                            } else {
                                for (auto node : postOp->outputs()[0]->targets()) {
                                    auto it = std::find(node->inputs().begin(), node->inputs().end(), postOp->outputs()[0]);
                                    node->reconnect(node->inputs()[std::distance(node->inputs().begin(), it)], opMatch->outputs()[0]);
                                }
                            }
                            g->internal().eraseNode(postOp);
                        }
                    }
                }
            }
            return true;
        };
    };

}// namespace refactor::computation
#endif// COMPUTATION_MATMUL_TRANSPOSE_H
