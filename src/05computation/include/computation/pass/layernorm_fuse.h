#ifndef COMPUTATION_LAYERNORM_FUSE_H
#define COMPUTATION_LAYERNORM_FUSE_H

#include "../graph.h"
#include "computation/operators/layernorm.h"
#include "computation/operators/reduce.h"
#include "computation/operators/simple_binary.h"
#include "computation/operators/simple_unary.h"
#include "computation/pass/converter.h"

namespace refactor::computation {

    class LayernormFuse : public Converter {
    public:
        virtual bool execute(const std::shared_ptr<GraphMutant> &g) const override {
            auto nodesList = g->internal().nodes();
            size_t count = 0;
            for (auto opMatch : nodesList) {
                if (opMatch->info().op == nullptr) {
                    continue;
                }
                size_t optype = opMatch->info().op->opTypeId();
                if (optype != SimpleBinary::typeId(refactor::kernel::SimpleBinaryType::Add)) {
                    continue;
                }
                if (opMatch->successors().size() < 2) {
                    continue;
                }
                auto input = opMatch->inputs()[0]->info().tensor;
                auto targets = opMatch->outputs()[0]->targets();
                auto ReduceMeanOp = *targets.begin();
                auto SubOp1 = *(std::next(targets.begin()));
                if (ReduceMeanOp == nullptr || SubOp1 == nullptr ||
                    ReduceMeanOp->info().op->opTypeId() != Reduce::typeId(refactor::kernel::ReduceType::Mean) ||
                    SubOp1->info().op->opTypeId() != SimpleBinary::typeId(refactor::kernel::SimpleBinaryType::Sub)) {
                    continue;
                }
                auto reduceOp = dynamic_cast<Reduce *>(ReduceMeanOp->info().op.get());
                auto axes = reduceOp->axes;
                if (axes.size() != 1) {
                    continue;
                }
                auto keepDims = reduceOp->keepDims;
                if (ReduceMeanOp->successors().size() != 1 || *(ReduceMeanOp->outputs()[0]->targets().begin()) != SubOp1) {
                    continue;
                }
                if (SubOp1->successors().size() != 2) {
                    continue;
                }
                auto targets1 = SubOp1->outputs()[0]->targets();
                auto PowOp = *targets1.begin();
                auto DivOp = *(std::next(targets1.begin()));
                if (PowOp == nullptr || DivOp == nullptr ||
                    PowOp->info().op->opTypeId() != SimpleBinary::typeId(refactor::kernel::SimpleBinaryType::Pow) ||
                    DivOp->info().op->opTypeId() != SimpleBinary::typeId(refactor::kernel::SimpleBinaryType::Div)) {
                    continue;
                }
                if (PowOp->successors().size() != 1 || DivOp->successors().size() != 1) {
                    continue;
                }
                auto ReduceMeanOp1 = *(PowOp->outputs()[0]->targets().begin());
                auto MulOp = *(DivOp->outputs()[0]->targets().begin());
                if (ReduceMeanOp1 == nullptr || MulOp == nullptr ||
                    ReduceMeanOp1->info().op->opTypeId() != Reduce::typeId(refactor::kernel::ReduceType::Mean) ||
                    MulOp->info().op->opTypeId() != SimpleBinary::typeId(refactor::kernel::SimpleBinaryType::Mul)) {
                    continue;
                }
                auto reduce1Op = dynamic_cast<Reduce *>(ReduceMeanOp1->info().op.get());
                auto axes1 = reduce1Op->axes;
                if (axes != axes1) {
                    continue;
                }
                if (auto keepDims1 = reduce1Op->keepDims; keepDims != keepDims1) {
                    continue;
                }
                if (MulOp->successors().size() != 1 || ReduceMeanOp1->successors().size() != 1) {
                    continue;
                }
                auto AddOrSqrtOp = *(ReduceMeanOp1->outputs()[0]->targets().begin());
                auto AddOp2 = *(MulOp->outputs()[0]->targets().begin());
                if (AddOrSqrtOp == nullptr || AddOp2 == nullptr ||
                    AddOp2->info().op->opTypeId() != SimpleBinary::typeId(refactor::kernel::SimpleBinaryType::Add)) {
                    continue;
                }
                if (AddOrSqrtOp->successors().size() != 1) {
                    continue;
                }
                float epsilon = 0.0;
                if (auto AddOp = AddOrSqrtOp; AddOp->info().op->opTypeId() == SimpleBinary::typeId(refactor::kernel::SimpleBinaryType::Add)) {
                    auto SqrtOp = *(AddOp->outputs()[0]->targets().begin());
                    if (SqrtOp == nullptr || SqrtOp->info().op->opTypeId() != SimpleUnary::typeId(refactor::kernel::SimpleUnaryType::Sqrt)) {
                        continue;
                    }
                    if (SqrtOp->successors().size() != 1 || *(SqrtOp->outputs()[0]->targets().begin()) != DivOp) {
                        continue;
                    }
                    // start replace with LayernormOp
                    if (auto t = AddOp->inputs()[1]->info().tensor->data; t) {
                        auto dtype = AddOp->inputs()[1]->info().tensor->dataType;
                        if (dtype == DataType::F32) {
                            epsilon = *t->get<float>();
                        } else if (dtype == DataType::FP16) {
                            epsilon = (*t->get<fp16_t>()).to_f32();
                        } else {
                            epsilon = 0.0;
                        }
                    }
                } else if (auto SqrtOp = AddOrSqrtOp; SqrtOp->info().op->opTypeId() == SimpleUnary::typeId(refactor::kernel::SimpleUnaryType::Sqrt)) {
                    if (*(SqrtOp->outputs()[0]->targets().begin()) != DivOp) {
                        continue;
                    }
                } else {
                    continue;
                }

                int axis = axes[0];
                auto layernormOp = g->internal().pushNode(
                    {std::make_unique<LayerNormalization>(epsilon, axis), fmt::format("Layernorm_{}", count)},
                    {g->internal().shareEdge({Tensor::share(input->dataType, input->shape), fmt::format("Layernorm_{}_out", count)})});
                layernormOp->connect(0, opMatch->outputs()[0]);
                layernormOp->connect(1, MulOp->inputs()[1]);
                layernormOp->connect(2, AddOp2->inputs()[1]);
                if (AddOp2->outputs()[0]->targets().size() == 0) {//global output
                    g->internal().replaceOutput(AddOp2->outputs()[0], layernormOp->outputs()[0]);
                } else {
                    for (auto node : AddOp2->outputs()[0]->targets()) {
                        auto it = std::find(node->inputs().begin(), node->inputs().end(), AddOp2->outputs()[0]);
                        node->reconnect(node->inputs()[std::distance(node->inputs().begin(), it)], layernormOp->outputs()[0]);
                    }
                }
                count++;
                g->internal().cleanup();
            }
            return true;
        };
    };


}// namespace refactor::computation

#endif// COMPUTATION_LAYERNORM_FUSE_H
