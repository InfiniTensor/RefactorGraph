#include "computation/operators/pool.h"
#include "common.h"
#include "common/range.h"

namespace refactor::onnx {
    using namespace common;

    InferResult inferPool(Operator const &op, TensorRefs inputs, InferOptions const &) {
        EXPECT_SIZE(1)

        auto const &input = inputs[0];
        if (!input.dataType.isIeee754()) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }

        auto const &kernelShape = op.attribute("kernel_shape").ints();
        if (input.rank() != kernelShape.size() + 2) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        SmallInts<4> input_(input.rank() - 2);
        for (auto i : range(2l, input.rank())) {
            EXPECT_VAL(input.shape[i], d)
            input_[i - 2] = d;
        }

        OptionalInts
            dilations = std::nullopt,
            pads = std::nullopt,
            strides = std::nullopt;
        if (auto it = op.attributes.find("dilations"); it != op.attributes.end()) {
            dilations.emplace(it->second.ints());
        }
        if (auto it = op.attributes.find("pads"); it != op.attributes.end()) {
            pads.emplace(it->second.ints());
        }
        if (auto it = op.attributes.find("strides"); it != op.attributes.end()) {
            strides.emplace(it->second.ints());
        }

        auto res = pool(input_, kernelShape, dilations, pads, strides);
        if (res.isErr()) {
            return Err(InferError(ERROR_MSG(res.unwrapErr())));
        }
        auto output_ = std::move(res.unwrap());
        Shape output(input.rank(), DimExpr(0));
        output[0] = input.shape[0];
        output[1] = input.shape[1];
        std::copy(output_.begin(), output_.end(), output.begin() + 2);
        return Ok(Tensors{Tensor::share(input.dataType, std::move(output), extractDependency(inputs))});
    }

    static computation::PoolType unsupport(OpType opType) {
        RUNTIME_ERROR(fmt::format("{} not support in unary lowering", opType.name()));
    }

    LowerOperator lowerPool(Operator const &op, TensorRefs) {
        using namespace computation;

        auto type = op.opType.is("onnx::AveragePool") ? PoolType::Average
                    : op.opType.is("onnx::LpPool")    ? PoolType::Lp
                    : op.opType.is("onnx::MaxPool")   ? PoolType::Max
                                                      : unsupport(op.opType);
        return {std::make_shared<Pool>(type), {0}};
    }

}// namespace refactor::onnx
