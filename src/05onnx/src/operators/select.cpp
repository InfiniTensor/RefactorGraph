#include "computation/operators/select.h"
#include "common.h"
#include "common/range.h"
#include <execution>

namespace refactor::onnx {
    using namespace common;

    InferResult inferSelect(Operator const &op, TensorRefs inputs, InferOptions const &options) {
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
        auto ans = Tensor::share(dataType, std::move(output), extractDependency(inputs));
        if (!options.shouldCalculate(inputs, {*ans})) {
            return Ok(Tensors{std::move(ans)});
        }

        auto opType = op.opType.name();
        auto eleSize = dataType.size();
        auto dst = reinterpret_cast<uint8_t *>(ans->malloc());

        //-------------------------------------
#define CASE(T)                                                                                                              \
    case DataType::T: {                                                                                                      \
        using t = primitive_t<DataType::T>::type;                                                                            \
        std::vector<t> src(inputs.size());                                                                                   \
        for (auto i : range0_(ans->elementsSize())) {                                                                        \
            auto indices = locateN(ans->shape, i);                                                                           \
            auto dst_ = reinterpret_cast<t *>(dst + i * eleSize);                                                            \
            std::transform(inputs.begin(), inputs.end(), src.begin(),                                                        \
                           [&indices](auto const &input) { return *reinterpret_cast<t const *>(locate1(input, indices)); }); \
            if (opType == "onnx::Min") {                                                                                     \
                *dst_ = *std::min_element(src.begin(), src.end());                                                           \
            } else {                                                                                                         \
                *dst_ = *std::max_element(src.begin(), src.end());                                                           \
            }                                                                                                                \
        }                                                                                                                    \
    } break;
        //-------------------------------------

        switch (inputs[0].dataType.internal) {
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
                ans->free();
                break;
        }

        return Ok(Tensors{std::move(ans)});
    }

    LowerOperator lowerSelect(Operator const &op, TensorRefs inputs) {
        using namespace computation;

        auto type = op.opType.is("onnx::Max")   ? SelectType::Max
                    : op.opType.is("onnx::Min") ? SelectType::Min
                                                : UNREACHABLEX(SelectType,
                                                               "{} not support",
                                                               op.opType.name());
        decltype(LowerOperator::inputs) inputs_(inputs.size());
        std::iota(inputs_.begin(), inputs_.end(), 0);
        return {std::make_shared<Select>(type), std::move(inputs_)};
    }
}// namespace refactor::onnx
