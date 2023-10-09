#include "constant.hh"
#include "common.h"

namespace refactor::onnx {
    using namespace common;
    using Op = Constant;

    Op::Constant(Attribute value_)
        : Operator(), value(std::move(value_)) {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        Attribute value;
        if (auto it = attributes.find("value"); it != attributes.end()) {
            value = std::move(it->second);
        } else if (auto it = attributes.find("value_float"); it != attributes.end()) {
            value = std::move(it->second);
        } else if (auto it = attributes.find("value_floats"); it != attributes.end()) {
            value = std::move(it->second);
        } else if (auto it = attributes.find("value_int"); it != attributes.end()) {
            value = std::move(it->second);
        } else if (auto it = attributes.find("value_ints"); it != attributes.end()) {
            value = std::move(it->second);
        } else if (auto it = attributes.find("value_string"); it != attributes.end()) {
            value = std::move(it->second);
        } else if (auto it = attributes.find("value_strings"); it != attributes.end()) {
            value = std::move(it->second);
        } else {
            RUNTIME_ERROR("Constant value not support");
        }
        return OpBox(std::make_unique<Op>(std::move(value)));
    }

    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Constant"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(0)

        if (value.isTensor()) {
            return Ok(Tensors{value.tensor()});
        }
        if (value.isFloat()) {
            auto ans = Tensor::share(DataType::F32, {}, {});
            *reinterpret_cast<float *>(ans->malloc()) = value.float_();
            return Ok(Tensors{std::move(ans)});
        }
        if (value.isFloats()) {
            auto const &x = value.floats();
            auto ans = Tensor::share(DataType::F32, {DimExpr(x.size())}, {});
            std::copy(x.begin(), x.end(), reinterpret_cast<float *>(ans->malloc()));
            return Ok(Tensors{std::move(ans)});
        }
        if (value.isInt()) {
            auto ans = Tensor::share(DataType::I64, Shape{}, {});
            *reinterpret_cast<int64_t *>(ans->malloc()) = value.int_();
            return Ok(Tensors{std::move(ans)});
        }
        if (value.isInts()) {
            auto const &x = value.ints();
            auto ans = Tensor::share(DataType::I64, {DimExpr(x.size())}, {});
            std::copy(x.begin(), x.end(), reinterpret_cast<int64_t *>(ans->malloc()));
            return Ok(Tensors{std::move(ans)});
        }
        return Err(InferError(ERROR_MSG("Constant value not support")));
    }
}// namespace refactor::onnx
