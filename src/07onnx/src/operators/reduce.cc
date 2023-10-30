#include "computation/operators/reduce.h"
#include "common.h"
#include "computation/operators/identity.h"
#include "reduce.hh"
#include <execution>

namespace refactor::onnx {
    using Op = Reduce;
    using Ty = ReduceType;

    Op::Reduce(Ty type_, bool keepdims_, bool noopWithEmptyAxes_)
        : Operator(),
          type(type_),
          keepdims(keepdims_),
          noopWithEmptyAxes(noopWithEmptyAxes_) {}

    auto Op::build(std::string_view opType, Attributes attributes) -> OpBox {
        auto keepDims = defaultOr(attributes, "keepdims", {1}).int_() != 0;
        auto noopWithEmptyAxes = defaultOr(attributes, "noop_with_empty_axes", {0}).int_() != 0;
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
        return OpBox(std::make_unique<Op>(ty, keepDims, noopWithEmptyAxes));
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
        if (inputs.size() == 1) {
            if (noopWithEmptyAxes) {
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
        if (axes.dataType != DataType::I64 || axes.rank() != 1 || !axes.data) {
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
        if (inputs.size() == 1) {
            if (noopWithEmptyAxes) {
                return std::make_unique<computation::Identity>();
            } else {
                return std::make_unique<Op_>(type_, decltype(Op_::axes){}, rank, keepdims);
            }
        }
        auto const &axes = inputs[1];
        auto axes_ = axes.data->get<int64_t>();
        auto axesSize = axes.shape[0].value();

        decltype(Op_::axes) axes__(axesSize);
        std::transform(std::execution::unseq,
                       axes_, axes_ + axesSize, axes__.begin(), [rank](auto axis) {
                           return axis < 0 ? axis + rank : axis;
                       });

        return std::make_unique<Op_>(type_, std::move(axes__), rank, keepdims);
    }
}// namespace refactor::onnx
