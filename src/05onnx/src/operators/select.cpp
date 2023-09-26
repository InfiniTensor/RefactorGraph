#include "computation/operators/select.h"
#include "common.h"

namespace refactor::onnx {
    using namespace common;

    InferResult inferSelect(Operator const &op, TensorRefs inputs, InferOptions options) {
        if (inputs.empty()) {
            return Err(InferError(ERROR_MSG("Input size error")));
        }
        ShapeRefs shapes;
        shapes.reserve(inputs.size());
        auto dataType = inputs[0].dataType;
        for (auto &input : inputs) {
            if (input.dataType != dataType) {
                return Err(InferError(ERROR_MSG("Input data type not support")));
            }
            shapes.emplace_back(input.shape);
        }
        MULTIDIR_BROADCAST(shapes)
        return Ok(Tensors{Tensor::share(dataType, std::move(output), extractDependency(inputs))});
    }

    static computation::SelectType unsupport(OpType opType) {
        RUNTIME_ERROR(fmt::format("{} not support in select lowering", opType.name()));
    }

    computation::SharedOp lowerSelect(Operator const &op, TensorRefs) {
        using namespace computation;

        auto type = op.opType.is("onnx::Max")   ? SelectType::Max
                    : op.opType.is("onnx::Min") ? SelectType::Min
                                                : unsupport(op.opType);
        return std::make_shared<Select>(type);
    }
}// namespace refactor::onnx
