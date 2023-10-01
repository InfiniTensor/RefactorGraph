﻿#include "computation/operators/reduce.h"
#include "common.h"
#include "common/range.h"
#include "computation/operators/identity.h"
#include <execution>

namespace refactor::onnx {
    using namespace common;

    InferResult inferReduce(Operator const &op, TensorRefs inputs, InferOptions const &options) {
        if (inputs.empty() || 2 < inputs.size()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        auto const &data = inputs[0];
        if (!data.dataType.isNumberic()) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        auto keepdims = op.attribute("keepdims", {1}).int_() != 0;
        if (inputs.size() == 1) {
            if (op.attribute("noop_with_empty_axes", {0}).int_() != 0) {
                return Ok(Tensors{Tensor::share(data)});
            } else if (keepdims) {
                return Ok(Tensors{Tensor::share(data.dataType,
                                                Shape(data.rank(), DimExpr(1)),
                                                extractDependency(inputs))});
            } else {
                return Ok(Tensors{Tensor::share(data.dataType,
                                                Shape{},
                                                extractDependency(inputs))});
            }
        }
        auto const &axes = inputs[1];
        if (axes.dataType != DataType::I64 || axes.rank() != 1 || !axes.hasData()) {
            return Err(InferError(ERROR_MSG("Axes not support")));
        }
        auto axes_ = axes.data->get<int64_t>();
        auto const &shape = data.shape;
        EXPECT_VAL(axes.shape[0], axesSize)
        std::unordered_set<int64_t> axes__;
        for (auto i : range0_(axesSize)) {
            auto axis = axes_[i];
            axes__.insert(axis < 0 ? axis + shape.size() : axis);
        }
        Shape output;
        for (auto i : range0_(shape.size())) {
            if (axes__.find(i) == axes__.end()) {
                output.emplace_back(shape[i]);
            } else if (keepdims) {
                output.emplace_back(1);
            }
        }
        return Ok(Tensors{Tensor::share(data.dataType,
                                        std::move(output),
                                        extractDependency(inputs))});
    }

    static computation::ReduceType unsupport(OpType opType) {
        RUNTIME_ERROR(fmt::format("{} not support in reduce lowering", opType.name()));
    }

    LowerOperator lowerReduce(Operator const &op, TensorRefs inputs) {
        using namespace computation;

        auto type = op.opType.is("onnx::ReduceMean")        ? ReduceType::Mean
                    : op.opType.is("onnx::ReduceL1")        ? ReduceType::L1
                    : op.opType.is("onnx::ReduceL2")        ? ReduceType::L2
                    : op.opType.is("onnx::ReduceLogSum")    ? ReduceType::LogSum
                    : op.opType.is("onnx::ReduceLogSumExp") ? ReduceType::LogSumExp
                    : op.opType.is("onnx::ReduceMax")       ? ReduceType::Max
                    : op.opType.is("onnx::ReduceMin")       ? ReduceType::Min
                    : op.opType.is("onnx::ReduceProd")      ? ReduceType::Prod
                    : op.opType.is("onnx::ReduceSum")       ? ReduceType::Sum
                    : op.opType.is("onnx::ReduceSumSquare") ? ReduceType::SumSquare
                                                            : unsupport(op.opType);

        auto rank = inputs[0].rank();
        auto keepdims = op.attribute("keepdims", {1}).int_() != 0;
        if (inputs.size() == 1) {
            if (op.attribute("noop_with_empty_axes", {0}).int_() != 0) {
                return {std::make_shared<Identity>(), {0}};
            } else {
                return {std::make_shared<Reduce>(type, decltype(Reduce::axes){}, keepdims), {0}};
            }
        }
        auto const &axes = inputs[1];
        auto axes_ = axes.data->get<int64_t>();
        auto axesSize = axes.shape[0].value();

        decltype(Reduce::axes) axes__(axesSize);
        std::transform(std::execution::unseq,
                       axes_, axes_ + axesSize, axes__.begin(), [rank](auto axis) {
                           return axis < 0 ? axis + rank : axis;
                       });

        return {std::make_shared<Reduce>(type, std::move(axes__), keepdims), {0}};
    }
}// namespace refactor::onnx