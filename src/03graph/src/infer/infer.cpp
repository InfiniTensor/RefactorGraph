#include "infer.h"
#include "data_type.h"
#include "error_handler.h"

using namespace refactor::common;

namespace refactor::graph {
    /// @brief 检查输入边是否是指定的数量。
    template<class T>
    inline void checkInputSize(std::vector<T> const &inputs, len_t size) {
        if (inputs.size() != size) {
            RUNTIME_ERROR("invalid inputs");
        }
    }

    inline std::vector<LayoutDim> multidirBroadcast(std::vector<std::vector<LayoutDim>> const &inputs) {
        std::vector<LayoutDim> ans;
        TODO("做个迭代器");
        return ans;
    }

    std::vector<EdgeInfo> inferAbs(std::vector<EdgeInfo> inputs) {
        checkInputSize(inputs, 1);
        if (isNumbericDataType(inputs[0].tensor().dataType)) {
            return inputs;
        } else {
            RUNTIME_ERROR("data type not support");
        }
    }

    std::vector<EdgeInfo> inferTrigonometry(std::vector<EdgeInfo> inputs) {
        checkInputSize(inputs, 1);
        if (isIeee754DataType(inputs[0].tensor().dataType)) {
            return inputs;
        } else {
            RUNTIME_ERROR("data type not support");
        }
    }

    std::vector<EdgeInfo> inferTanh(std::vector<EdgeInfo> inputs) {
        checkInputSize(inputs, 1);
        if (isFloatDataType(inputs[0].tensor().dataType)) {
            return inputs;
        } else {
            RUNTIME_ERROR("data type not support");
        }
    }

    std::vector<EdgeInfo> inferArithmetic(std::vector<EdgeInfo> inputs) {
        checkInputSize(inputs, 2);
        if (inputs[0].isTensor()) {
            auto i0 = inputs[0].tensor();
            auto i1 = inputs[1].tensor();
            if (isNumbericDataType(i0.dataType) && i0.dataType == i1.dataType) {
                return {EdgeInfo{Tensor{i0.dataType, multidirBroadcast({i0.layout, i1.layout})}}};
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
