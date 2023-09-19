#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferConstant(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(0) {
            if (auto it = op.attributes.find("value"); it != op.attributes.end()) {
                return Ok(Tensors{it->second.tensor()});
            }
            if (auto it = op.attributes.find("value_float"); it != op.attributes.end()) {
                auto value = it->second.float_();
                auto ans = Tensor::share(DataType::F32, {});
                *reinterpret_cast<float *>(ans->malloc()) = value;
                return Ok(Tensors{std::move(ans)});
            }
            if (auto it = op.attributes.find("value_floats"); it != op.attributes.end()) {
                auto const &value = it->second.floats();
                auto ans = Tensor::share(DataType::F32, {DimExpr(value.size())});
                std::copy(value.begin(), value.end(), reinterpret_cast<float *>(ans->malloc()));
                return Ok(Tensors{std::move(ans)});
            }
            if (auto it = op.attributes.find("value_int"); it != op.attributes.end()) {
                auto value = it->second.int_();
                auto ans = Tensor::share(DataType::I64, Shape{});
                *reinterpret_cast<int64_t *>(ans->malloc()) = value;
                return Ok(Tensors{std::move(ans)});
            }
            if (auto it = op.attributes.find("value_ints"); it != op.attributes.end()) {
                auto const &value = it->second.ints();
                auto ans = Tensor::share(DataType::I64, {DimExpr(value.size())});
                std::copy(value.begin(), value.end(), reinterpret_cast<int64_t *>(ans->malloc()));
                return Ok(Tensors{std::move(ans)});
            }
            return Err(InferError(ERROR_MSG("Constant value not support")));
        }
    }
}// namespace refactor::onnx
