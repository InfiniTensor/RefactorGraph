#include "computation/operators/reshape.h"
#include "common.h"
#include "refactor/common.h"
#include "reshape.hh"

namespace refactor::onnx {
    using Op = Reshape;

    Op::Reshape(bool allowzero_)
        : Operator(), allowzero(allowzero_) {}

    auto Op::build(std::string_view, Attributes attributes) -> OpBox {
        auto allowzero = defaultOr(attributes, "allowzero", {0}).int_() != 0;
        return OpBox(std::make_unique<Op>(allowzero));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Reshape"; }
    auto Op::valueDependentInputs() const -> InputVec { return {1}; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        EXPECT_SIZE(2)

        auto const &data = inputs[0];
        auto const &shape = inputs[1];
        if (shape.dataType != DataType::I64 || shape.rank() != 1 || !shape.data) {
            return Err(InferError(ERROR_MSG("Shape not support")));
        }

        ASSERT(!allowzero, "Not support allowzero");

        auto shape_ = shape.data->get<int64_t>();
        EXPECT_VAL(shape.shape[0], rank)

        Shape output(rank, DimExpr(1));
        int pos_1 = -1, mulOld = 1, mul = 1;
        auto it = data.shape.begin();
        for (auto i : range0_(static_cast<size_t>(rank))) {
            if (shape_[i] == -1) {
                if (pos_1 != -1) {
                    return Err(InferError(ERROR_MSG("Invalid shape value")));
                }
                pos_1 = i;
                if (it != data.shape.end()) {
                    auto const &d = *it++;
                    EXPECT_VAL(d, v)
                    mulOld *= v;
                }
            } else if (shape_[i] == 0) {
                if (it == data.shape.end()) {
                    return Err(InferError(ERROR_MSG("Invalid shape value")));
                }
                auto const &d = *it++;
                output[i] = d;
            } else {
                output[i] = DimExpr(shape_[i]);
                mul *= shape_[i];

                if (it != data.shape.end()) {
                    auto const &d = *it++;
                    EXPECT_VAL(d, v)
                    mulOld *= v;
                }
            }
        }
        while (it != data.shape.end()) {
            auto const &d = *it++;
            EXPECT_VAL(d, v)
            mulOld *= v;
        }

        if (pos_1 != -1) {
            auto div = std::div(mulOld, mul);
            if (div.rem != 0) {
                return Err(InferError(ERROR_MSG("Invalid shape value")));
            } else {
                output[pos_1] = DimExpr(div.quot);
            }
        } else if (mulOld != mul) {
            return Err(InferError(ERROR_MSG("Invalid shape value")));
        }
        return Ok(Tensors{Tensor::share(data.dataType, std::move(output), extractDependency(inputs), data.data)});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::Reshape;
        return std::make_unique<Op_>();
    }

}// namespace refactor::onnx
