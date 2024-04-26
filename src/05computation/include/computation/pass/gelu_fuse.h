#ifndef COMPUTATION_GELU_FUSE_H
#define COMPUTATION_GELU_FUSE_H

#include "../graph.h"
#include "computation/operators/gelu.h"
#include "computation/operators/reshape.h"
#include "computation/operators/simple_binary.h"
#include "computation/operators/simple_unary.h"
#include "computation/pass/converter.h"

namespace refactor::computation {
    class GeluFuse : public Converter {
    public:
        virtual bool execute(const std::shared_ptr<GraphMutant> &g) const override {
            auto nodesList = g->internal().nodes();
            size_t count = 0;
            for (auto opMatch : nodesList) {
                if (opMatch->info().op == nullptr) {
                    continue;
                }
                size_t optype = opMatch->info().op->opTypeId();
                if (optype != SimpleBinary::typeId(refactor::kernel::SimpleBinaryType::Add) &&
                    optype != Reshape::typeId()) {
                    continue;
                }
                auto input = opMatch->outputs()[0]->info().tensor;
                auto targets = opMatch->outputs()[0]->targets();
                if (opMatch->successors().size() >= 3) {

                } else if (opMatch->successors().size() >= 2) {
                    // op1 is Div op2 is Mul
                    auto op1 = *targets.begin();
                    auto op2 = *(std::next(targets.begin()));
                    if (op1 == nullptr || op2 == nullptr ||
                        op1->info().op->opTypeId() != SimpleBinary::typeId(refactor::kernel::SimpleBinaryType::Div) ||
                        op2->info().op->opTypeId() != SimpleBinary::typeId(refactor::kernel::SimpleBinaryType::Mul)) {
                        continue;
                    }
                    if (op1->successors().size() != 1 || op2->successors().size() != 1) {
                        continue;
                    }
                    auto ErfOp = *(op1->outputs()[0]->targets().begin());
                    auto MulOp = *(op2->outputs()[0]->targets().begin());
                    if (ErfOp == nullptr || MulOp == nullptr ||
                        ErfOp->info().op->opTypeId() != SimpleUnary::typeId(refactor::kernel::SimpleUnaryType::Erf) ||
                        MulOp->info().op->opTypeId() != SimpleBinary::typeId(refactor::kernel::SimpleBinaryType::Mul)) {
                        continue;
                    }
                    if (auto alpha = MulOp->inputs()[1]->info().tensor->data; alpha) {
                        float alphaVal = *alpha->get<float>();
                        if (alphaVal != 0.5f) {
                            continue;
                        }
                    } else {
                        continue;
                    }
                    if (ErfOp->successors().size() != 1) {
                        continue;
                    }
                    auto AddOp = *(ErfOp->outputs()[0]->targets().begin());
                    if (AddOp == nullptr || AddOp->info().op->opTypeId() != SimpleBinary::typeId(refactor::kernel::SimpleBinaryType::Add)) {
                        continue;
                    }
                    if (auto beta = AddOp->inputs()[1]->info().tensor->data; beta) {
                        float betaVal = *beta->get<float>();
                        if (betaVal != 1.0f) {
                            continue;
                        }
                    } else {
                        continue;
                    }
                    if (AddOp->successors().size() != 1 || *(AddOp->outputs()[0]->targets().begin()) != op2) {
                        continue;
                    }
                    // replace
                    auto geluOp = g->internal().pushNode(
                        {std::make_unique<Gelu>(), fmt::format("Gelu_{}", count)},
                        {g->internal().shareEdge({Tensor::share(input->dataType, input->shape), fmt::format("Gelu_{}_out", count)})});
                    geluOp->connect(0, opMatch->outputs()[0]);
                    if (MulOp->outputs()[0]->targets().size() == 0) {
                        g->internal().replaceOutput(MulOp->outputs()[0], geluOp->outputs()[0]);
                    } else {
                        for (auto node : MulOp->outputs()[0]->targets()) {
                            auto it = std::find(node->inputs().begin(), node->inputs().end(), MulOp->outputs()[0]);
                            node->reconnect(node->inputs()[std::distance(node->inputs().begin(), it)], geluOp->outputs()[0]);
                        }
                    }
                    count++;
                    g->internal().cleanup();
                }
            }
            return true;
        };
    };
}// namespace refactor::computation

#endif//COMPUTATION_GELU_FUSE_H