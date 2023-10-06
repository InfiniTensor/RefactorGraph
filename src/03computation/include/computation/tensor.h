#ifndef COMPUTATION_TENSOR_H
#define COMPUTATION_TENSOR_H

#include "common/data_type.h"
#include "layout.h"
#include "mem_manager/blob.h"
#include <absl/container/inlined_vector.h>
#include <string>

namespace refactor::computation {

    using Shape = absl::InlinedVector<int64_t, 4>;

    struct Tensor {
        common::DataType dataType;
        Shape shape;
        LayoutType layout;
        std::shared_ptr<mem_manager::Blob> data;

        Tensor(common::DataType,
               Shape,
               LayoutType,
               std::shared_ptr<mem_manager::Blob>) noexcept;
        static std::shared_ptr<Tensor>
            share(common::DataType,
                  Shape,
                  LayoutType = LayoutType::Others,
                  std::shared_ptr<mem_manager::Blob> = nullptr) noexcept;
    };

    struct Edge {
        std::shared_ptr<Tensor> tensor;
        std::string name;
    };

}// namespace refactor::computation

#endif// COMPUTATION_TENSOR_H
