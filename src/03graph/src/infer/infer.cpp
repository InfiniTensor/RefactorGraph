#include "infer.h"
#include "common/data_type.h"
#include "common/error_handler.h"

using namespace refactor::common;

namespace refactor::graph {
    /// @brief 多方向形状广播。
    /// @param inputs 所有输入的形状。
    /// @return 广播后的形状。
    inline static Shape multidirBroadcast(std::vector<Shape> const &inputs) {
        Shape ans;
        for (auto i = 0;; ++i) {
            auto any = false;
            len_t value = 1;
            for (auto const &input : inputs) {
                if (i < input.size()) {
                    any = true;
                    if (value == 1) {
                        value = input[i];
                    } else if (input[i] != 1 && input[i] != value) {
                        RUNTIME_ERROR("invalid broadcast");
                    }
                }
            }
            if (any) {
                ans.push_back(value);
            } else {
                break;
            }
        }
        ans.shrink_to_fit();
        return ans;
    }

    InferError::InferError(std::string &&msg)
        : std::runtime_error(std::forward<std::string>(msg)) {}

    InferResult inferAbs(Edges inputs) {
        if (inputs.size() != 1) {
            return Err(INFER_ERROR("Input size error"));
        } else if (!isNumbericDataType(inputs[0].tensor().dataType)) {
            return Err(INFER_ERROR("Data type not support"));
        } else {
            return Ok(std::move(inputs));
        }
    }

    InferResult inferTrigonometry(Edges inputs) {
        if (inputs.size() != 1) {
            return Err(INFER_ERROR("Input size error"));
        } else if (!isIeee754DataType(inputs[0].tensor().dataType)) {
            return Err(INFER_ERROR("Data type not support"));
        } else {
            return Ok(std::move(inputs));
        }
    }

    InferResult inferTanh(Edges inputs) {
        if (inputs.size() != 1) {
            return Err(INFER_ERROR("Input size error"));
        } else if (!isFloatDataType(inputs[0].tensor().dataType)) {
            return Err(INFER_ERROR("Data type not support"));
        } else {
            return Ok(std::move(inputs));
        }
    }

    InferResult inferArithmetic(Edges inputs) {
        if (inputs.size() != 2) {
            return Err(INFER_ERROR("Input size error"));
        } else if (inputs[0].isTensor()) {
            auto i0 = inputs[0].tensor();
            auto i1 = inputs[1].tensor();
            if (isNumbericDataType(i0.dataType) && i0.dataType == i1.dataType) {
                auto ans = Tensor{i0.dataType, multidirBroadcast({std::move(i0.shape), std::move(i1.shape)})};
                return Ok(Edges{EdgeInfo{std::move(ans)}});
            } else {
                return Err(INFER_ERROR("Data type not support"));
            }
        } else {
            auto i0 = inputs[0].shapeVariable();
            auto i1 = inputs[1].shapeVariable();
            TODO("calculate shape variable");
        }
    }


}// namespace refactor::graph
