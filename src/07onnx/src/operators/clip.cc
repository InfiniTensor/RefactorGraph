#include "clip.hh"
#include "common.h"
#include "computation/operators/clip.h"

namespace refactor::onnx {
    using Op = Clip;

    auto Op::build(ModelContext const &ctx, std::string_view, Attributes) -> OpBox {
        auto iter = ctx.find("opset_version");
        auto opsetVer = iter != ctx.end() ? iter->second.int_() : StandardOpsetVersion;
        if (opsetVer <= 6) {
            TODO("Not support yet");
        }
        return OpBox(std::make_unique<Op>());
    }
    auto Op::typeId() -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto Op::opTypeId() const -> size_t { return typeId(); }
    auto Op::opTypeName() const -> std::string_view { return "onnx::Clip"; }

    auto Op::infer(TensorRefs inputs, InferOptions const &options) const -> InferResult {
        if (inputs.size() < 1 || 3 < inputs.size()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }

        auto const &input = inputs[0];
        if (!input.dataType.isNumberic()) {
            return Err(InferError(ERROR_MSG("Input data type not support")));
        }
        return Ok(Tensors{Tensor::share(input.dataType, input.shape, extractDependency(inputs))});
    }

    auto Op::lower(TensorRefs) const -> computation::OpBox {
        using Op_ = computation::Clip;
        return std::make_unique<Op_>();
    }

}// namespace refactor::onnx
