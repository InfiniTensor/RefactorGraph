#include "all_reduce.hh"
#include "common.h"
#include "computation/operators/all_reduce.h"

namespace refactor::communication {
    using Op = AllReduce;
    using Ty = kernel::AllReduceType;

    Op::AllReduce(Ty type_) : Operator(), type(type_) {}

    auto Op::build(ModelContext const &, std::string_view opType, Attributes attributes) -> OpBox {
        // clang-format off
        auto type_ =
            opType == "onnx::AllReduceAvg"    ? Ty::Avg   :
            opType == "onnx::AllReduceSum"    ? Ty::Sum   :
            opType == "onnx::AllReduceMin"    ? Ty::Min   :
            opType == "onnx::AllReduceMax"    ? Ty::Max   :
            opType == "onnx::AllReduceProd"   ? Ty::Prod  :
            UNREACHABLEX(Ty, "Unsupported allReduce operator: {}", opType);
        // clang-format on
        return OpBox(std::make_unique<Op>(type_));
    }

    auto Op::typeId(Ty type_) -> size_t {
        switch (type_) {
            case Ty::Sum: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Avg: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Min: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Max: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case Ty::Prod: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }

    auto Op::opTypeId() const -> size_t { return typeId(type); }
    auto Op::opTypeName() const -> std::string_view {
        switch (type) {
            case Ty::Sum:
                return "AllReduceSum";
            case Ty::Avg:
                return "AllReduceAvg";
            case Ty::Min:
                return "AllReduceMin";
            case Ty::Max:
                return "AllReduceMax";
            case Ty::Prod:
                return "AllReduceProd";
            default:
                UNREACHABLE();
        }
    }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(1)

        return Ok(Tensors{Tensor::share(inputs[0].dataType,
                                        inputs[0].shape,
                                        extractDependency(inputs))});
    }

    computation::OpBox Op::lower(TensorRefs inputs) const {

        return std::make_unique<computation::AllReduce>(type);
    }

}// namespace refactor::communication
