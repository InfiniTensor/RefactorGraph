#include "computation/operators/cast.h"
#include "common.h"
#include "common/natural.h"
#include <execution>

namespace refactor::onnx {
    using namespace common;

    template<class TS, class TD>
    void castData(void const *src, void *dst, size_t size) {
        auto src_ = reinterpret_cast<TS const *>(src);
        auto dst_ = reinterpret_cast<TD *>(dst);
        std::transform(std::execution::unseq, src_, src_ + size, dst_, [](auto x) { return static_cast<TD>(x); });
    }

    InferResult inferCast(Operator const &op, TensorRefs inputs, InferOptions const &options) {
        EXPECT_SIZE(1)

        auto const &input = inputs[0];
        auto to = *DataType::parse(op.attribute("to").int_());
        auto ans = Tensor::share(to, input.shape, extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans})) {
            return Ok(Tensors{std::move(ans)});
        }
        auto from = input.dataType;
        if (from == to) {
            ans->data = input.data;
            return Ok(Tensors{std::move(ans)});
        }
        auto size = ans->elementsSize();
        auto src = input.data->get<void>();
        auto dst = ans->malloc();
        switch (from.internal) {
            case DataType::F32:
                switch (to.internal) {
                    case DataType::I32:
                        castData<float, int32_t>(src, dst, size);
                        break;

                    case DataType::I64:
                        castData<float, int64_t>(src, dst, size);
                        break;

                    case DataType::Bool: {
                        auto src_ = reinterpret_cast<float const *>(src);
                        auto dst_ = reinterpret_cast<bool *>(dst);
                        std::transform(std::execution::unseq, src_, src_ + size, dst_, [](auto x) { return x != 0.0; });
                        break;
                    }

                    default:
                        break;
                }
                break;

            case DataType::I32:
                switch (to.internal) {
                    case DataType::F32:
                        castData<int32_t, float>(src, dst, size);
                        break;

                    case DataType::I64:
                        castData<int32_t, int64_t>(src, dst, size);
                        break;

                    case DataType::Bool: {
                        auto src_ = reinterpret_cast<int32_t const *>(src);
                        auto dst_ = reinterpret_cast<bool *>(dst);
                        std::transform(std::execution::unseq, src_, src_ + size, dst_, [](auto x) { return x != 0; });
                        break;
                    }

                    default:
                        break;
                }
                break;

            case DataType::I64:
                switch (to.internal) {
                    case DataType::F32:
                        castData<int64_t, float>(src, dst, size);
                        break;

                    case DataType::I32:
                        castData<int64_t, int32_t>(src, dst, size);
                        break;

                    case DataType::Bool: {
                        auto src_ = reinterpret_cast<int64_t const *>(src);
                        auto dst_ = reinterpret_cast<bool *>(dst);
                        std::transform(std::execution::unseq, src_, src_ + size, dst_, [](auto x) { return x != 0; });
                        break;
                    }

                    default:
                        break;
                }
                break;

            default:
                break;
        }
        return Ok(Tensors{std::move(ans)});
    }

    LowerOperator lowerCast(Operator const &op, TensorRefs) {
        using namespace computation;

        auto to = *DataType::parse(op.attribute("to").int_());
        return {std::make_shared<Cast>(to), {0}};
    }
}// namespace refactor::onnx
