#include "infer.h"
#include "data_type.h"

using namespace refactor::common;

namespace refactor::graph {
    template<class T>
    inline void checkInputSize(std::vector<T> const &inputs, len_t size) {
        if (inputs.size() != size) {
            RUNTIME_ERROR("invalid inputs");
        }
    }

    std::vector<EdgeInfo> inferAbs(std::vector<EdgeInfo> inputs) {
        checkInputSize(inputs, 1);
        if (!isNumbericDataType(inputs[0].tensor().dataType)) {
            return inputs;
        } else {
            RUNTIME_ERROR("data type not support");
        }
    }

    std::vector<EdgeInfo> inferTrigonometry(std::vector<EdgeInfo> inputs) {
        checkInputSize(inputs, 1);
        if (!isIeee754DataType(inputs[0].tensor().dataType)) {
            return inputs;
        } else {
            RUNTIME_ERROR("data type not support");
        }
    }

    std::vector<EdgeInfo> inferTanh(std::vector<EdgeInfo> inputs) {
        checkInputSize(inputs, 1);
        if (!isFloatDataType(inputs[0].tensor().dataType)) {
            return inputs;
        } else {
            RUNTIME_ERROR("data type not support");
        }
    }

}// namespace refactor::graph
