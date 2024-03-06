#ifndef COMPUTATION_CONV_TO_MATMUL_H
#define COMPUTATION_CONV_TO_MATMUL_H

#include "../graph.h"
#include "computation/operators/conv.h"
#include "computation/operators/mat_mul.h"
#include "computation/operators/reshape.h"
#include "computation/operators/transpose.h"
#include "computation/pass/converter.h"

namespace refactor::computation {
    class ConvToMatmul : public Converter {

    public:
        /*
         *   input         weight
         *     |             |
         *     |             |
         *   transpose     transpose
         *     |             |
         *     |             |
         *   reshape      reshape  
         *        \       /
         *         \     /
         *          matmul
         *            |
         *          reshape
         *            |
         *          transpose
         *            |
         *          output
         */
        virtual bool execute(const std::shared_ptr<GraphMutant> &g) const override {
            auto nodesList = g->internal().nodes();
            size_t count = 0;
            for (auto opMatch : nodesList) {
                if (opMatch->info().op == nullptr) {
                    continue;
                }
                size_t optype = opMatch->info().op->opTypeId();
                if (optype != Conv::typeId()) {
                    continue;
                }
                auto convOp = dynamic_cast<Conv *>(opMatch->info().op.get());
                auto input = opMatch->inputs()[0]->info().tensor;
                auto weight = opMatch->inputs()[1]->info().tensor;
                auto shape = weight->shape;
                // judge conv is 1x1 convolution
                if (shape.size() != 4 || shape[2] != 1 || shape[3] != 1) {
                    continue;
                }
                auto attr = convOp->attributes;
                auto poolAttrRank = attr.rank();
                auto poolAttrDilation = attr.dilations();
                auto poolAttrStride = attr.strides();
                auto poolAttrPad = attr.pads();
                bool flag = false;
                for (auto i : range0_(poolAttrRank)) {
                    if (poolAttrDilation[i] != 1 || poolAttrStride[i] != 1) {
                        flag = true;
                        break;
                    }
                    if (poolAttrPad[i] != 0 || poolAttrPad[i + poolAttrRank] != 0) {
                        flag = true;
                        break;
                    }
                }
                if (flag) { continue; }
                // create transpose op
                absl::InlinedVector<uint32_t, 4>
                    perm1 = {0, 2, 3, 1};
                Shape shape1 = {input->shape[0], input->shape[2], input->shape[3], input->shape[1]};
                auto newTransposeOp1 = g->internal().pushNode(
                    {std::make_unique<Transpose>(perm1), fmt::format("ConvToMatmul_transpose1_{}", count)},
                    {g->internal().shareEdge({Tensor::share(input->dataType, shape1), fmt::format("ConvToMatmul_transpose1_{}_out", count)})});
                newTransposeOp1->connect(0, opMatch->inputs()[0]);
                absl::InlinedVector<uint32_t, 4> perm2 = {1, 0, 2, 3};
                Shape shape2 = {weight->shape[1], weight->shape[0], weight->shape[2], weight->shape[3]};
                auto newTransposeOp2 = g->internal().pushNode(
                    {std::make_unique<Transpose>(perm2), fmt::format("ConvToMatmul_transpose2_{}", count)},
                    {g->internal().shareEdge({Tensor::share(weight->dataType, shape2), fmt::format("ConvToMatmul_transpose2_{}_out", count)})});
                newTransposeOp2->connect(0, opMatch->inputs()[1]);
                // create reshape op
                Shape shape3 = {input->shape[0] * input->shape[2] * input->shape[3], input->shape[1]};
                Shape shape4 = {weight->shape[1], weight->shape[0]};
                int64_t data1[2] = {input->shape[0] * input->shape[2] * input->shape[3], input->shape[1]};
                int64_t data2[2] = {weight->shape[1], weight->shape[0]};
                auto [data1_, ptr1] = refactor::kernel::Blob::share(sizeof(int64_t) * 2);
                auto [data2_, ptr2] = refactor::kernel::Blob::share(sizeof(int64_t) * 2);
                ptr1 = &data1[0];
                ptr2 = &data2[0];
                auto newReshapeEdge1 = g->internal().shareEdge({Tensor::share(DataType::I64, {2}, LayoutType::Others, data1_), fmt::format("ConvToMatmul_reshape1_shape_{}", count)});
                auto newReshapeEdge2 = g->internal().shareEdge({Tensor::share(DataType::I64, {2}, LayoutType::Others, data2_), fmt::format("ConvToMatmul_reshape2_shape_{}", count)});
                auto newReshapeOp1 = g->internal().pushNode(
                    {std::make_unique<Reshape>(), fmt::format("ConvToMatmul_reshape1_{}", count)},
                    {g->internal().shareEdge({Tensor::share(input->dataType, shape3), fmt::format("ConvToMatmul_reshape1_{}_out", count)})});
                auto newReshapeOp2 = g->internal().pushNode(
                    {std::make_unique<Reshape>(), fmt::format("ConvToMatmul_reshape2_{}", count)},
                    {g->internal().shareEdge({Tensor::share(weight->dataType, shape4), fmt::format("ConvToMatmul_reshape2_{}_out", count)})});
                newReshapeOp1->connect(0, newTransposeOp1->outputs()[0]);
                newReshapeOp1->connect(1, newReshapeEdge1);
                newReshapeOp2->connect(0, newTransposeOp2->outputs()[0]);
                newReshapeOp2->connect(1, newReshapeEdge2);
                // create matmul op
                Shape shape5 = {input->shape[0] * input->shape[2] * input->shape[3], weight->shape[0]};
                auto newMatMulOp = g->internal().pushNode(
                    {std::make_unique<MatMul>(1.0, 1.0, false, false), fmt::format("ConvToMatmul_matmul_{}", count)},
                    {g->internal().shareEdge({Tensor::share(input->dataType, shape5), fmt::format("ConvToMatmul_matmul_{}_out", count)})});
                newMatMulOp->connect(0, newReshapeOp1->outputs()[0]);
                newMatMulOp->connect(1, newReshapeOp2->outputs()[0]);
                // create reshape op
                Shape shape6 = {input->shape[0], input->shape[2], input->shape[3], weight->shape[0]};
                int64_t data3[4] = {input->shape[0], input->shape[2], input->shape[3], weight->shape[0]};
                auto [data3_, ptr3] = refactor::kernel::Blob::share(sizeof(int64_t) * 4);
                ptr3 = &data3[0];
                auto newReshapeEdge3 = g->internal().shareEdge({Tensor::share(DataType::I64, {4}, LayoutType::Others, data3_), fmt::format("ConvToMatmul_reshape3_shape_{}", count)});
                auto newReshapeOp3 = g->internal().pushNode(
                    {std::make_unique<Reshape>(), fmt::format("ConvToMatmul_reshape3_{}", count)},
                    {g->internal().shareEdge({Tensor::share(input->dataType, shape6), fmt::format("ConvToMatmul_reshape3_{}_out", count)})});
                newReshapeOp3->connect(0, newMatMulOp->outputs()[0]);
                newReshapeOp3->connect(1, newReshapeEdge3);
                // create transpose op
                absl::InlinedVector<uint32_t, 4> perm3 = {0, 3, 1, 2};
                Shape shape7 = {input->shape[0], weight->shape[0], input->shape[2], input->shape[3]};
                auto newTransposeOp3 = g->internal().pushNode(
                    {std::make_unique<Transpose>(perm3), fmt::format("ConvToMatmul_transpose3_{}", count)},
                    {g->internal().shareEdge({Tensor::share(input->dataType, shape7), fmt::format("ConvToMatmul_transpose3_{}_out", count)})});
                newTransposeOp3->connect(0, newReshapeOp3->outputs()[0]);
                if (opMatch->outputs()[0]->targets().size() == 0) {// global output
                    g->internal().replaceOutput(opMatch->outputs()[0], newTransposeOp3->outputs()[0]);
                } else {
                    for (auto node : opMatch->outputs()[0]->targets()) {
                        auto it = std::find(node->inputs().begin(), node->inputs().end(), opMatch->outputs()[0]);
                        node->reconnect(node->inputs()[std::distance(node->inputs().begin(), it)], newTransposeOp3->outputs()[0]);
                    }
                }
                g->internal().eraseNode(opMatch);
                count++;
            }
            return true;
        };
    };

}// namespace refactor::computation
#endif// COMPUTATION_CONV_TO_MATMUL_H