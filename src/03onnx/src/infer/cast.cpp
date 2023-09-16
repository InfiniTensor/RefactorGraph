#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    template<class TS, class TD>
    void castData(void *src, void *dst, size_t size) {
        auto src_ = reinterpret_cast<TS *>(src);
        auto dst_ = reinterpret_cast<TD *>(dst);
        std::transform(src_, src_ + size, dst_, [](auto x) { return static_cast<TD>(x); });
    }

    InferResult inferCast(Operator const &op, Tensors inputs) {
        EXPECT_SIZE(1) {
            auto const &input = inputs[0];
            auto to = static_cast<DataType>(op.attribute("to").int_());
            auto ans = Tensor::share(to, input->shape);
            if (!shouldCalculate(inputs, ans->shape)) {
                return Ok(Tensors{std::move(ans)});
            }
            auto from = input->dataType;
            if (from == to) {
                ans->data = input->data;
                return Ok(Tensors{std::move(ans)});
            }
            auto size = ans->elementsSize();
            auto src = input->data->ptr;
            auto dst = ans->malloc();
            switch (from) {
                case DataType::F32:
                    switch (to) {
                        case DataType::I64:
                            castData<float, int64_t>(src, dst, size);
                            break;

                        case DataType::Bool: {
                            auto src_ = reinterpret_cast<float *>(src);
                            auto dst_ = reinterpret_cast<bool *>(dst);
                            std::transform(src_, src_ + size, dst_, [](auto x) { return x != 0.0; });
                            break;
                        }

                        default:
                            break;
                    }
                    break;

                case DataType::I64:
                    switch (to) {
                        case DataType::F32:
                            castData<int64_t, float>(src, dst, size);
                            break;

                        case DataType::Bool: {
                            auto src_ = reinterpret_cast<int64_t *>(src);
                            auto dst_ = reinterpret_cast<bool *>(dst);
                            std::transform(src_, src_ + size, dst_, [](auto x) { return x != 0; });
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
    }
}// namespace refactor::onnx
