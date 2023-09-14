#include "infer.h"
#include <numeric>

namespace refactor::onnx {
    using namespace refactor::common;

    template<DataType T>
    void accumulate_(void *dst, void *src, void *acc) {
        using T_ = typename primitive_t<T>::type;
        *reinterpret_cast<T_ *>(dst) = *reinterpret_cast<T_ *>(src) + *reinterpret_cast<T_ *>(acc);
    }

    void accumulate(DataType dataType, void *dst, void *src, size_t stepBytes) {
        auto acc = reinterpret_cast<uint8_t *>(src) - stepBytes;
        switch (dataType) {
#define CASE(T)                                  \
    case DataType::T:                            \
        accumulate_<DataType::T>(dst, src, acc); \
        break;
            CASE(F32);
            CASE(F64);
            CASE(I32);
            CASE(I64);
            CASE(I8);
            CASE(I16);
            CASE(U8);
            CASE(U16);
            CASE(U32);
            CASE(U64);
            default:
                TODO("DataType not support");
        }
    }

    InferResult inferCumSum(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(2) {
            auto const &x = inputs[0];
            auto const &axis = inputs[1];
            if (!axis->shape.empty()) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            auto dataType = x->dataType;
            if (!isNumbericDataType(dataType) ||
                (axis->dataType != DataType::I64 &&
                 axis->dataType != DataType::I32)) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            auto output = x->shape;
            if (!shouldCalculate(inputs, output)) {
                return Ok(Tensors{Tensor::share(dataType, std::move(output))});
            }
            auto exclusive = op.attribute("exclusive", {0}).int_();
            auto reverse = op.attribute("reverse", {0}).int_() != 0;
            if (reverse) {// TODO: support reverse
                return Ok(Tensors{Tensor::share(dataType, std::move(output))});
            }
            auto axis_ = axis->dataType == DataType::I64
                             ? *reinterpret_cast<int64_t *>(axis->data->ptr)
                             : *reinterpret_cast<int32_t *>(axis->data->ptr);
            if (axis_ < 0) {
                axis_ += output.size();
            }
            if (axis_ < 0 || output.size() <= axis_) {
                return Err(InferError(ERROR_MSG("Invalid axis")));
            }
            auto size = sizeOf(output);
            auto eleSize = dataTypeSize(dataType);
            auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
            auto src = reinterpret_cast<uint8_t *>(x->data->ptr);
            auto dst = reinterpret_cast<uint8_t *>(blob->ptr);
            auto step = std::accumulate(output.begin() + axis_ + 1, output.end(), eleSize,
                                        [](auto const acc, auto const &d) { return acc * d.value(); });
            if (!reverse) {
                for (size_t i = 0; i < size; ++i) {
                    auto indices = locateN(output, i);
                    auto axisIdx = indices[axis_];
                    auto dst_ = dst + i * eleSize;
                    if (axisIdx == 0) {
                        if (exclusive) {
                            std::memset(dst_, 0, eleSize);
                        } else {
                            std::memcpy(dst_, src + i * eleSize, eleSize);
                        }
                    } else {
                        auto src_ = src + i * eleSize;
                        if (exclusive) {
                            accumulate(dataType, dst_, src_ - step, step);
                        } else {
                            accumulate(dataType, dst_, src_, step);
                        }
                    }
                }
            } else {
                UNREACHABLE();
            }
            return Ok(Tensors{Tensor::share(dataType, std::move(output), std::move(blob))});
        }
    }
}// namespace refactor::onnx
