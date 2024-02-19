#include "pad.hh"
#include "common.h"
#include "computation/operators/pad.h"
#include <execution>

namespace refactor::onnx {
    using Op = Pad;
    using Pm = PadMode;

    Op::Pad(Pm mode_) : Operator(), mode(mode_) {}

    auto Op::build(ModelContext const &, std::string_view opType, Attributes attributes) -> OpBox {
        auto mode = attributes.getOrInsert("mode", {"constant"}).string();
        Pm pm;
        if (mode == "constant") {
            pm = Pm::Constant;
        } else if (mode == "reflect") {
            pm = Pm::Reflect;
        } else if (mode == "edge") {
            pm = Pm::Edge;
        } else if (mode == "wrap") {
            pm = Pm::Wrap;
        } else {
            UNREACHABLEX(void, "Unsupported Pad mode: {}", mode);
        }
        return OpBox(std::make_unique<Op>(pm));
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Pad"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &) const -> InferResult {
        if (inputs.empty() || inputs.size() > 4 || inputs.size() < 2) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        auto const &input = inputs[0];
        auto const &pad = inputs[1];
        if (pad.dataType != DataType::I64 || pad.rank() != 1) {
            return Err(InferError(ERROR_MSG("Pad inputs pads is invalid")));
        }
        EXPECT_VAL(pad.shape[0], pad_len)
        if (!pad.data) {
            return Err(InferError(ERROR_MSG("Pad inputs pads must be constant")));
        }
        int64_t const *pads = pad.data->get<int64_t>();
        Ints pads_;
        // TODO: onnx padOp inputs pads support negative numbers
        for (auto i : range0_(pad.shape[0].value())) {
            if (auto pad_value = pads[i]; pad_value < 0) {
                return Err(InferError(ERROR_MSG("Pad inputs pads is not support negative numbers")));
            }
            pads_.push_back(pads[i]);
        }
        auto rank = input.rank();
        if (inputs.size() >= 3) {
            auto const &constant_value = inputs[2];
            if (constant_value.dataType != input.dataType) {
                return Err(InferError(ERROR_MSG("Pad inputs constant type not support")));
            }
        }
        if (inputs.size() == 4) {
            Ints pads__(2 * rank, 0);
            auto const &axes = inputs[3];
            if ((axes.dataType != DataType::I32 && axes.dataType != DataType::I64) || axes.rank() != 1) {
                return Err(InferError(ERROR_MSG("Pad inputs axes is invalid")));
            }
            if (!axes.data) {
                return Err(InferError(ERROR_MSG("Pad inputs axes must be constant")));
            }
            EXPECT_VAL(axes.shape[0], axes_len)
            if (pad_len != 2 * axes_len) {
                return Err(InferError(ERROR_MSG("Pad inputs pads len is not 2x axes")));
            }
            void const *axes_data = axes.data->get<void>();
            for (auto i : range0_(axes_len)) {
                auto axis = axes.dataType == DataType::I32 ? reinterpret_cast<int32_t const *>(axes_data)[i] : reinterpret_cast<int64_t const *>(axes_data)[i];
                if (axis < 0) {
                    axis += rank;
                }
                if (axis < 0 || axis >= rank) {
                    return Err(InferError(ERROR_MSG("Axes not support")));
                }
                pads__[axis] = pads_[i];
                pads__[axis + rank] = pads_[i + axes_len];
            }
            pads_ = pads__;
        } else {
            if (pad_len != 2 * rank) {
                return Err(InferError(ERROR_MSG("Pad inputs pads len is not 2x input")));
            }
        }
        Shape output_shape(rank, DimExpr(1));
        for (auto i : range0_(rank)) {
            output_shape[i] = DimExpr(input.shape[i].value() + pads_[i] + pads_[i + rank]);
        }
        auto ans = Tensor::share(input.dataType, output_shape, extractDependency(inputs));
        return Ok(Tensors{std::move(ans)});
    }

    auto Op::lower(TensorRefs inputs) const -> computation::OpBox {
        using Ty_ = computation::PadType;
        using Op_ = computation::Pad;
        using Dimension = computation::Dimensions;

        auto rank = inputs[0].rank();
        int64_t const *pads_ = inputs[1].data->get<int64_t>();
        std::vector<int64_t> pads_info(2 * rank, 0);
        Dimension dims(rank);
        if (inputs.size() != 4) {
            for (auto i : range0_(inputs[1].shape[0].value())) { pads_info[i] = pads_[i]; }
        } else {
            auto const &axes_ = inputs[3];
            void const *axes_data = axes_.data->get<void>();
            auto axes_len = axes_.shape[0].value();

            for (auto i : range0_(axes_len)) {
                auto axis = axes_.dataType == DataType::I32 ? reinterpret_cast<int32_t const *>(axes_data)[i] : reinterpret_cast<int64_t const *>(axes_data)[i];
                if (axis < 0) { axis += rank; }
                pads_info[axis] = pads_[i];
                pads_info[axis + rank] = pads_[i + axes_len];
            }
        }
        for (auto i : range0_(rank)) {
            auto dimI = inputs[0].shape[i].value();
            dims[i] = {
                dimI, dimI + pads_info[i] + pads_info[i + rank], pads_info[i]};
        }
        Ty_ mode_;
        switch (mode) {
            case Pm::Constant:
                mode_ = Ty_::Constant;
                break;
            case Pm::Reflect:
                mode_ = Ty_::Reflect;
                break;
            case Pm::Edge:
                mode_ = Ty_::Edge;
                break;
            case Pm::Wrap:
                mode_ = Ty_::Wrap;
                break;
            default:
                UNREACHABLE();
        }
        return std::make_unique<Op_>(std::move(dims), mode_);
    }

}// namespace refactor::onnx
