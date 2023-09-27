#ifndef COMPUTATION_TENSOR_H
#define COMPUTATION_TENSOR_H

#include "common/blob.h"
#include "common/data_type.h"
#include <absl/container/inlined_vector.h>
#include <string>

namespace refactor::computation {

    using Shape = absl::InlinedVector<int64_t, 4>;

    struct Tensor {
        common::DataType type;
        Shape shape;
        std::shared_ptr<common::Blob> data;
    };

    struct Edge {
        std::shared_ptr<Tensor> tensor;
        std::string name;
    };

}// namespace refactor::computation

#endif// COMPUTATION_TENSOR_H
