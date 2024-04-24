#include "depth_to_space.hh"
#include "common.h"
#include "computation/operators/depth_to_space.h"

namespace refactor::onnx {
    using Op = DepthToSpace;

    Op::DepthToSpace(Int blocksize_, String mode_)
        : Operator(), blocksize(blocksize_), mode(std::move(mode_)) {}

    auto Op::build(ModelContext const &, std::string_view, Attributes attributes) -> OpBox {
        auto blocksize = attributes["blocksize"].int_();
        auto mode = attributes.getOrInsert("mode", {"DCR"}).string();
        return OpBox(std::make_unique<Op>(blocksize, mode));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::DepthToSpace"; }
    auto Op::valueDependentInputs() const -> InputVec { return {1}; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        EXPECT_SIZE(1)

        if (mode != "DCR" && mode != "CRD") {
            return Err(InferError(ERROR_MSG("Invalid mode for DepthToSpace operator")));
        }

        auto const &input = inputs[0];
        if (input.rank() != 4) {
            return Err(InferError(ERROR_MSG("Input layout should be NCHW")));
        }
        auto n = input.shape[0];
        EXPECT_VAL(input.shape[1], c)
        EXPECT_VAL(input.shape[2], h)
        EXPECT_VAL(input.shape[3], w)
        return Ok(Tensors{Tensor::share(
            input.dataType,
            Shape{n, DimExpr(c / blocksize / blocksize), DimExpr(h * blocksize), DimExpr(w * blocksize)},
            extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::DepthToSpace;
        return std::make_unique<Op_>(blocksize, mode == "DCR" ? computation::DepthToSpaceMode::DCR : computation::DepthToSpaceMode::CRD);
    }

}// namespace refactor::onnx
