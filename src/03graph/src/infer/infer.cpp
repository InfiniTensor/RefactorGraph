#include "infer.h"
#include "common/data_type.h"
#include "common/error_handler.h"

using namespace refactor::common;

namespace refactor::graph {
    /// @brief 检查输入边是否是指定的数量。
    template<class T>
    inline void checkInputSize(std::vector<T> const &inputs, len_t size) {
        if (inputs.size() != size) {
            RUNTIME_ERROR("invalid inputs");
        }
    }

    inline std::vector<len_t> multidirBroadcast(std::vector<std::vector<len_t>> const &inputs) {
        std::vector<len_t> ans;
        TODO("做个迭代器");
        return ans;
    }

    InferError::InferError(std::string &&msg)
        : std::runtime_error(std::forward<std::string>(msg)) {}

    InferResult inferAbs(Edges inputs) {
        checkInputSize(inputs, 1);
        if (isNumbericDataType(inputs[0].tensor().dataType)) {
            return {Ok(std::move(inputs))};
        } else {
            RUNTIME_ERROR("data type not support");
        }
    }

    InferResult inferTrigonometry(Edges inputs) {
        checkInputSize(inputs, 1);
        if (isIeee754DataType(inputs[0].tensor().dataType)) {
            return {Ok(std::move(inputs))};
        } else {
            RUNTIME_ERROR("data type not support");
        }
    }

    InferResult inferTanh(Edges inputs) {
        checkInputSize(inputs, 1);
        if (isFloatDataType(inputs[0].tensor().dataType)) {
            return {Ok(std::move(inputs))};
        } else {
            RUNTIME_ERROR("data type not support");
        }
    }

    InferResult inferArithmetic(Edges inputs) {
        checkInputSize(inputs, 2);
        if (inputs[0].isTensor()) {
            auto i0 = inputs[0].tensor();
            auto i1 = inputs[1].tensor();
            if (isNumbericDataType(i0.dataType) && i0.dataType == i1.dataType) {
                auto ans = Tensor{i0.dataType, multidirBroadcast({std::move(i0.shape), std::move(i1.shape)})};
                return {Ok(Edges{EdgeInfo{std::move(ans)}})};
            } else {
                RUNTIME_ERROR("data type not support");
            }
        } else {
            auto i0 = inputs[0].shapeVariable();
            auto i1 = inputs[1].shapeVariable();
            TODO("calculate shape variable");
        }
    }


}// namespace refactor::graph
