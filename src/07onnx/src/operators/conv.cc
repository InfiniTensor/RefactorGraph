#include "computation/operators/conv.h"
#include "conv.hh"
#include "common.h"
#include <numeric>

namespace refactor::onnx {
    using Op = Conv;

    Op::Conv(OptionalInts dilations_,
             OptionalInts pads_,
             OptionalInts strides_)
        : Operator(),
          dilations(std::move(dilations_)),
          pads(std::move(pads_)),
          strides(std::move(strides_)) {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        OptionalInts
            dilations = std::nullopt,
            pads = std::nullopt,
            strides = std::nullopt;
        if (auto it = attributes.find("dilations"); it != attributes.end()) {
            dilations.emplace(std::move(it->second.ints()));
        }
        if (auto it = attributes.find("pads"); it != attributes.end()) {
            pads.emplace(std::move(it->second.ints()));
        }
        if (auto it = attributes.find("strides"); it != attributes.end()) {
            strides.emplace(std::move(it->second.ints()));
        }
        return OpBox(std::make_unique<Op>(std::move(dilations), std::move(pads), std::move(strides)));
    }

    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Conv"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        if (auto size = inputs.size(); size < 2 || 3 < size) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }

        auto const &input = inputs[0];
        auto const &kernel = inputs[1];

        auto dataType = input.dataType;
        if (!dataType.isIeee754() || kernel.dataType != dataType) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        auto rank = input.rank();
        if (rank < 2 || rank != kernel.rank()) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        if (inputs.size() == 3) {
            auto const &bias = inputs[2];
            if (bias.dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            if (bias.rank() != 1 || bias.shape[0] != kernel.shape[0]) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
        }

        if (input.shape[1] != kernel.shape[1]) {
            EXPECT_VAL(input.shape[1], input1)
            EXPECT_VAL(kernel.shape[1], kernel1)
            auto div = std::div(input1, kernel1);
            fmt::println("input1: {}, kernel1: {}", input1, kernel1);
            if (div.rem != 0) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            ASSERT(div.quot == 1, "group conv not support");
        }

        SmallInts<4> input_(input.rank() - 2);
        for (auto i : range(2l, input.rank())) {
            EXPECT_VAL(input.shape[i], d)
            input_[i - 2] = d;
        }
        Ints kernel_(kernel.rank() - 2);
        for (auto i : range(2l, kernel.rank())) {
            EXPECT_VAL(kernel.shape[i], d)
            kernel_[i - 2] = d;
        }

        auto res = pool(input_, kernel_, dilations, pads, strides);
        if (res.isErr()) {
            return Err(InferError(ERROR_MSG(res.unwrapErr())));
        }
        auto output_ = std::move(res.unwrap());
        Shape output(input.rank(), DimExpr(0));
        output[0] = input.shape[0];
        output[1] = kernel.shape[0];
        std::copy(output_.begin(), output_.end(), output.begin() + 2);
        return Ok(Tensors{Tensor::share(input.dataType, std::move(output), extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Op_ = computation::Conv;

        auto rank = inputs[0].rank();
        return std::make_unique<Op_>(computation::PoolAttributes(
            rank - 2,
            dilations ? dilations->data() : nullptr,
            pads ? pads->data() : nullptr,
            strides ? strides->data() : nullptr));
    }

}// namespace refactor::onnx
