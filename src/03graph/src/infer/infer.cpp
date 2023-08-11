#include "infer.h"
#include "data_type.h"

using namespace refactor::common;

namespace refactor::graph {

    std::vector<EdgeInfo> inferAbs(std::vector<EdgeInfo> inputs) {
        if (inputs.size() != 1) {
            RUNTIME_ERROR("inputs.size() != 1");
        }
        if (std::holds_alternative<Tensor>(inputs[0])) {
            if (!isNumbericDataType(std::get<Tensor>(inputs[0]).dataType)) {
                return inputs;
            } else {
                RUNTIME_ERROR("data type not support");
            }
        } else {
            RUNTIME_ERROR("edge type not support");
        }
    }

}// namespace refactor::graph
