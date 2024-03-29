﻿#include "reduce.hh"
#include "common.h"
#include "computation/operators/identity.h"
#include <execution>

namespace refactor::onnx {
    using Op = Reduce;
    using Ty = ReduceType;

    Op::Reduce(Ty type_, decltype(axes) axes_, bool keepdims_, bool noopWithEmptyAxes_)
        : Operator(),
          type(type_),
          axes(std::move(axes_)),
          keepdims(keepdims_),
          noopWithEmptyAxes(noopWithEmptyAxes_) {}

    auto Op::build(ModelContext const &ctx, std::string_view opType, Attributes attributes) -> OpBox {
        auto iter = ctx.find("opset_version");
        auto opsetVer = iter != ctx.end() ? iter->second.int_() : StandardOpsetVersion;

        auto noopWithEmptyAxes = false;
        decltype(Op::axes) axes = std::nullopt;

        // 针对ReduceSum做特判
        if (opType == "onnx::ReduceSum") {
            if (opsetVer >= 13) {
                noopWithEmptyAxes = attributes.getOrInsert("noop_with_empty_axes", {0}).int_() != 0;
            } else {
                axes.emplace(attributes.getOrInsert("axes", {{}}).ints());
            }
        } else {
            if (opsetVer >= 18) {
                noopWithEmptyAxes = attributes.getOrInsert("noop_with_empty_axes", {0}).int_() != 0;
            } else {
                axes.emplace(attributes.getOrInsert("axes", {{}}).ints());
            }
        }

        auto keepDims = attributes.getOrInsert("keepdims", {1}).int_();
        Ty ty;
        if (opType == "onnx::ReduceMean") {
            ty = Ty::Mean;
        } else if (opType == "onnx::ReduceL1") {
            ty = Ty::L1;
        } else if (opType == "onnx::ReduceL2") {
            ty = Ty::L2;
        } else if (opType == "onnx::ReduceLogSum") {
            ty = Ty::LogSum;
        } else if (opType == "onnx::ReduceLogSumExp") {
            ty = Ty::LogSumExp;
        } else if (opType == "onnx::ReduceMax") {
            ty = Ty::Max;
        } else if (opType == "onnx::ReduceMin") {
            ty = Ty::Min;
        } else if (opType == "onnx::ReduceProd") {
            ty = Ty::Prod;
        } else if (opType == "onnx::ReduceSum") {
            ty = Ty::Sum;
        } else if (opType == "onnx::ReduceSumSquare") {
            ty = Ty::SumSquare;
        } else {
            UNREACHABLEX(void, "Unsupported reduce operator: {}", opType);
        }
        return OpBox(std::make_unique<Op>(ty, std::move(axes), keepDims, noopWithEmptyAxes));
    }

    auto Op::typeId(Ty type) -> size_t {
        switch (type) {
            case Ty::Mean: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::L1: {
                static uint8_t ID = 2;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::L2: {
                static uint8_t ID = 3;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::LogSum: {
                static uint8_t ID = 4;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::LogSumExp: {
                static uint8_t ID = 5;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Max: {
                static uint8_t ID = 6;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Min: {
                static uint8_t ID = 7;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Prod: {
                static uint8_t ID = 8;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Sum: {
                static uint8_t ID = 9;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::SumSquare: {
                static uint8_t ID = 10;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }

    auto Op::opTypeId() const -> size_t { return typeId(type); }

    auto Op::opTypeName() const -> std::string_view {
        switch (type) {
            case Ty::Mean:
                return "onnx::ReduceMean";
            case Ty::L1:
                return "onnx::ReduceL1";
            case Ty::L2:
                return "onnx::ReduceL2";
            case Ty::LogSum:
                return "onnx::ReduceSum";
            case Ty::LogSumExp:
                return "onnx::ReduceSumExp";
            case Ty::Max:
                return "onnx::ReduceMax";
            case Ty::Min:
                return "onnx::ReduceMin";
            case Ty::Prod:
                return "onnx::ReduceProd";
            case Ty::Sum:
                return "onnx::ReduceSum";
            case Ty::SumSquare:
                return "onnx::ReduceSumSquare";
            default:
                UNREACHABLE();
        }
    }

    auto Op::valueDependentInputs() const -> InputVec { return {1}; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        if (inputs.empty() || 2 < inputs.size()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        auto const &data = inputs[0];
        if (!data.dataType.isNumberic()) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        if (!axes && inputs.size() == 1) {
            return Ok(Tensors{
                noopWithEmptyAxes ? Tensor::share(data)
                : keepdims        ? Tensor::share(data.dataType, Shape(data.rank(), DimExpr(1)), extractDependency(inputs))
                                  : Tensor::share(data.dataType, Shape{}, extractDependency(inputs))});
        }
        std::span<int64_t const> axes_;
        if (axes) {
            axes_ = *axes;
        } else {
            auto const &axes__ = inputs[1];
            if (axes__.dataType != DataType::I64 ||
                axes__.rank() != 1 ||
                !axes__.data) {
                return Err(InferError(ERROR_MSG("Axes not support")));
            }
            EXPECT_VAL(axes__.shape[0], axesSize)
            axes_ = std::span(axes__.data->get<int64_t>(), axesSize);
        }
        auto const &shape = data.shape;
        std::unordered_set<int64_t> axes__;
        for (auto axis : axes_) {
            axes__.insert(axis < 0 ? axis + shape.size() : axis);
        }
        Shape output;
        for (auto i : range0_(shape.size())) {
            if (!axes__.contains(i)) {
                output.emplace_back(shape[i]);
            } else if (keepdims) {
                output.emplace_back(1);
            }
        }
        return Ok(Tensors{Tensor::share(data.dataType,
                                        std::move(output),
                                        extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Op_ = computation::Reduce;
        using Ty_ = computation::ReduceType;

        Ty_ type_;
        switch (type) {
            case Ty::Mean:
                type_ = Ty_::Mean;
                break;
            case Ty::L1:
                type_ = Ty_::L1;
                break;
            case Ty::L2:
                type_ = Ty_::L2;
                break;
            case Ty::LogSum:
                type_ = Ty_::LogSum;
                break;
            case Ty::LogSumExp:
                type_ = Ty_::LogSumExp;
                break;
            case Ty::Max:
                type_ = Ty_::Max;
                break;
            case Ty::Min:
                type_ = Ty_::Min;
                break;
            case Ty::Prod:
                type_ = Ty_::Prod;
                break;
            case Ty::Sum:
                type_ = Ty_::Sum;
                break;
            case Ty::SumSquare:
                type_ = Ty_::SumSquare;
                break;
            default:
                UNREACHABLE();
        }

        auto rank = inputs[0].rank();
        if (!axes && inputs.size() == 1) {
            if (noopWithEmptyAxes) {
                return std::make_unique<computation::Identity>();
            } else {
                decltype(Op_::axes) axes(rank);
                std::iota(axes.begin(), axes.end(), 0);
                return std::make_unique<Op_>(type_, std::move(axes), rank, keepdims);
            }
        }
        std::span<int64_t const> axes_;
        if (axes) {
            axes_ = *axes;
        } else {
            axes_ = std::span(inputs[1].data->get<int64_t>(), inputs[1].shape[0].value());
        }
        decltype(Op_::axes) axes__(axes_.size());
        std::transform(std::execution::unseq,
                       axes_.begin(), axes_.end(),
                       axes__.begin(),
                       [rank](auto axis) { return axis < 0 ? axis + rank : axis; });
        return std::make_unique<Op_>(type_, std::move(axes__), rank, keepdims);
    }
}// namespace refactor::onnx
