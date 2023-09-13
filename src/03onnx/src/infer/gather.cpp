#include "infer.h"

namespace refactor::onnx {
    using namespace refactor::common;

    InferResult inferGather(Operator const &op, Edges inputs) {
        EXPECT_SIZE(2)
        if (inputs[1]->dataType != DataType::I32 && inputs[1]->dataType != DataType::I64) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        } else {
            auto const r = inputs[0]->shape.size();
            auto const q = inputs[1]->shape.size();
            auto axis = op.attribute("axis", {0}).int_();
            if (axis < 0) {
                axis += r;
            }
            if (axis < 0 || r <= axis) {
                return Err(InferError(ERROR_MSG("Input shape not support")));
            }
            auto dataType = inputs[0]->dataType;
            auto ans = inputs[0]->shape;
            auto const &indices = inputs[1]->shape;
            ans.erase(ans.begin() + axis);
            ans.insert(ans.begin() + axis, indices.begin(), indices.end());
            if (!shouldCalculate(inputs, ans)) {
                return Ok(Edges{std::make_shared<Tensor>(dataType, std::move(ans))});
            }

            auto const ssz = ans.size();
            absl::InlinedVector<int64_t, 4> sX(r), sY(q);

            auto getter = [](DimExpr const &d) { return d.value(); };
            std::transform(inputs[0]->shape.begin(), inputs[0]->shape.end(), sX.begin(), getter);
            std::transform(inputs[1]->shape.begin(), inputs[1]->shape.end(), sY.begin(), getter);

            auto [sZ, size] = shape_size(ans);
            auto eleSize = dataTypeSize(inputs[0]->dataType);
            auto blob = std::make_shared<Blob>(new uint8_t[size * eleSize]);
            auto srcX = reinterpret_cast<uint8_t *>(inputs[0]->data->ptr);
            auto srcY = reinterpret_cast<int64_t *>(inputs[1]->data->ptr);
            auto dst = reinterpret_cast<uint8_t *>(blob->ptr);

            for (size_t i = 0; i < size; ++i) {
                std::vector<int64_t> indices(ssz);
                auto zi = i;
                auto it = indices.rbegin();
                for (auto d : sZ) {
                    auto div = std::div(zi, d);
                    *it++ = div.rem;
                    zi = div.quot;
                }

                size_t ii = 0, mul = 1;
                for (size_t j = 0; j < q; ++j) {
                    auto j_ = q - 1 - j;// reverse
                    ii += indices[j_] * mul;
                    mul *= sY[j_];
                }
                auto k = srcY[ii];

                ii = 0, mul = 1;
                for (size_t j = axis + q; j < ssz; ++j) {
                    auto j_ = ssz - 1 - j;// reverse
                    ii += indices[j_] * mul;
                    mul *= sX[j_ - q + 1];
                }
                ii += k * mul;
                for (size_t j = 0; j < axis; ++j) {
                    auto j_ = axis - 1 - j;// reverse
                    ii += indices[j_] * mul;
                    mul *= sX[j_];
                }

                std::copy_n(srcX + ii * eleSize, eleSize, dst + i * eleSize);
                fmt::println("gather copies {} bytes", eleSize);
            }

            return Ok(Edges{std::make_shared<Tensor>(dataType, std::move(ans), std::move(blob))});
        }
    }

}// namespace refactor::onnx
