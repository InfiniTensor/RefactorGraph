#ifndef KERNEL_TENSOR_H
#define KERNEL_TENSOR_H

#include "common/data_type.h"
#include "mem_manager/blob.hh"
#include <absl/container/inlined_vector.h>

namespace refactor::kernel {

    using Shape = absl::InlinedVector<int64_t, 4>;

    enum class LayoutType {
        NCHW,
        NHWC,
        Others,
    };

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

        bool hasData() const;
        int64_t rank() const;
        size_t elementsSize() const;
        size_t bytesSize() const;
    };

    using TensorRefs = absl::InlinedVector<std::reference_wrapper<Tensor const>, 2>;

}// namespace refactor::kernel

#endif// KERNEL_TENSOR_H
