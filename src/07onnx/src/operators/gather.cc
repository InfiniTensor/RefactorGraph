#include "computation/operators/gather.h"
#include "common.h"
#include "gather.hh"
#include "refactor/common.h"
#include <execution>

namespace refactor::onnx {
    using Op = Gather;

    Op::Gather(Int axis_)
        : Operator(), axis(axis_) {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        auto axis = defaultOr(attributes, "axis", {0}).int_();
        return OpBox(std::make_unique<Op>(axis));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Gather"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(2)

        auto const &data = inputs[0];
        auto const &indices = inputs[1];
        if (indices.dataType != DataType::I32 && indices.dataType != DataType::I64) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }

        auto const rank = data.rank();
        auto axis_ = axis < 0 ? axis + rank : axis;
        if (axis_ < 0 || rank <= axis_) {
            return Err(InferError(ERROR_MSG("Input shape not support")));
        }
        auto output = data.shape;
        output.erase(output.begin() + axis_);
        output.insert(output.begin() + axis_, indices.shape.begin(), indices.shape.end());
        auto ans = Tensor::share(data.dataType, std::move(output), extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans})) {
            return Ok(Tensors{std::move(ans)});
        }

        std::for_each_n(std::execution::unseq, natural_t(0), ans->elementsSize(),
                        [&data, &indices, &output,
                         axis_,
                         q = indices.shape.size(),
                         ssz = output.size(),
                         src = data.data->get<uint8_t>(),
                         dst = reinterpret_cast<uint8_t *>(ans->malloc()),
                         eleSize = data.dataType.size()](auto const i) {
                            auto indices_ = locateN(output, i);
                            int64_t k;
                            {
                                size_t ii = 0, mul = 1;
                                for (auto j : range0_(q).rev()) {
                                    ii += indices_[j] * mul;
                                    mul *= indices.shape[j].value();
                                }
                                k = indices.dataType == DataType::I64
                                        ? indices.data->get<int64_t>()[ii]
                                        : indices.data->get<int32_t>()[ii];
                            }
                            {
                                size_t ii = 0, mul = 1;
                                for (auto j : range(static_cast<decltype(q)>(axis_) + q, ssz).rev()) {
                                    ii += indices_[j] * mul;
                                    mul *= data.shape[j - q + 1].value();
                                }
                                ii += k * mul;
                                for (auto j : range0_(axis_).rev()) {
                                    ii += indices_[j] * mul;
                                    mul *= data.shape[j].value();
                                }
                                std::memcpy(dst + i * eleSize, src + ii * eleSize, eleSize);
                            }
                        });

        return Ok(Tensors{std::move(ans)});
    }

    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Op_ = computation::Gather;
        auto rank = inputs[0].rank();
        return std::make_unique<Op_>(axis < 0 ? axis + rank : axis, rank);
    }

}// namespace refactor::onnx
