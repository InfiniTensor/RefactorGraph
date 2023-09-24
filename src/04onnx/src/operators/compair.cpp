#include "computation/operators/compair.h"
#include "common.h"
#include "common/range.h"

namespace refactor::onnx {
    using namespace common;

    InferResult inferCompair(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(2)

        auto const &a = inputs[0];
        auto const &b = inputs[1];
        if (a->dataType != b->dataType) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }

        MULTIDIR_BROADCAST((ShapeRefs{a->shape, b->shape}))
        auto ans = Tensor::share(DataType::Bool, std::move(output), extractDependency(inputs));
        if (!shouldCalculate(inputs, ans->shape) || a->dataType != DataType::I64) {// TODO: support other data type
            return Ok(Tensors{std::move(ans)});
        }

        auto dst = reinterpret_cast<bool *>(ans->malloc());
        for (auto i : range0_(ans->elementsSize())) {
            auto indices = locateN(ans->shape, i);
            auto a_ = *reinterpret_cast<int64_t *>(locate1(*a, indices)),
                 b_ = *reinterpret_cast<int64_t *>(locate1(*b, indices));
            if (op.opType.is("onnx::Equal")) {
                dst[i] = a_ == b_;
            } else if (op.opType.is("onnx::Greater")) {
                dst[i] = a_ > b_;
            } else if (op.opType.is("onnx::GreaterOrEqual")) {
                dst[i] = a_ >= b_;
            } else if (op.opType.is("onnx::Less")) {
                dst[i] = a_ < b_;
            } else if (op.opType.is("onnx::LessOrEqual")) {
                dst[i] = a_ <= b_;
            } else {
                return Err(InferError(ERROR_MSG("OpType not support")));
            }
        }
        return Ok(Tensors{std::move(ans)});
    }

    static computation::CompairType unsupport(OpType opType) {
        RUNTIME_ERROR(fmt::format("{} not support in compair lowering", opType.name()));
    }

    computation::SharedOp lowerCompair(Operator const &op, TensorRefs) {
        using namespace computation;

        auto type = op.opType.is("onnx::Equal")            ? CompairType::EQ
                    : op.opType.is("onnx::Greater")        ? CompairType::GT
                    : op.opType.is("onnx::GreaterOrEqual") ? CompairType::GE
                    : op.opType.is("onnx::Less")           ? CompairType::LT
                    : op.opType.is("onnx::LessOrEqual")    ? CompairType::LE
                                                           : unsupport(op.opType);
        return std::make_shared<Compair>(type);
    }
}// namespace refactor::onnx
