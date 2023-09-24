#include "computation/operators/cum_sum.h"
#include "common.h"
#include "common/range.h"
#include <execution>
#include <numeric>

namespace refactor::onnx {
    using namespace refactor::common;

    template<decltype(DataType::internal) T>
    void accumulate_(void *dst, void *src, void *acc) {
        using T_ = typename primitive_t<T>::type;
        *reinterpret_cast<T_ *>(dst) = *reinterpret_cast<T_ *>(src) + *reinterpret_cast<T_ *>(acc);
    }

    void accumulate(DataType dataType, void *dst, void *src, size_t stepBytes) {
        auto acc = reinterpret_cast<uint8_t *>(dst) - stepBytes;
        switch (dataType.internal) {
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
        EXPECT_SIZE(2)

        auto const &x = inputs[0];
        auto const &axis = inputs[1];
        if (!axis->shape.empty()) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        auto dataType = x->dataType;
        if (!dataType.isNumberic() ||
            (axis->dataType != DataType::I64 &&
             axis->dataType != DataType::I32)) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        auto ans = Tensor::share(dataType, x->shape, extractDependency(inputs));
        if (!shouldCalculate(inputs, ans->shape)) {
            return Ok(Tensors{std::move(ans)});
        }
        auto exclusive = op.attribute("exclusive", {0}).int_();
        auto reverse = op.attribute("reverse", {0}).int_() != 0;
        if (reverse) {// TODO: support reverse
            return Ok(Tensors{std::move(ans)});
        }
        auto rank = ans->rank();
        auto axis_ = axis->dataType == DataType::I64
                         ? *reinterpret_cast<int64_t *>(axis->data->ptr)
                         : *reinterpret_cast<int32_t *>(axis->data->ptr);
        if (axis_ < 0) {
            axis_ += rank;
        }
        if (axis_ < 0 || rank <= axis_) {
            return Err(InferError(ERROR_MSG("Invalid axis")));
        }
        auto eleSize = dataType.size();
        if (!reverse) {
            std::for_each_n(std::execution::seq, natural_t(0), ans->elementsSize(),
                            [&, axis_, exclusive, eleSize,
                             step = std::accumulate(ans->shape.begin() + axis_ + 1, ans->shape.end(), eleSize,
                                                    [](auto const acc, auto const &d) { return acc * d.value(); }),
                             src = reinterpret_cast<uint8_t *>(x->data->ptr),
                             dst = reinterpret_cast<uint8_t *>(ans->malloc())](auto i) {
                                auto indices = locateN(ans->shape, i);
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
                            });
        } else {
            UNREACHABLE();
        }
        return Ok(Tensors{std::move(ans)});
    }

    computation::SharedOp lowerCumSum(Operator const &op, TensorRefs) {
        using namespace computation;

        auto exclusive = op.attribute("exclusive", {0}).int_() != 0;
        auto reverse = op.attribute("reverse", {0}).int_() != 0;
        return std::make_shared<CumSum>(exclusive, reverse);
    }
}// namespace refactor::onnx
