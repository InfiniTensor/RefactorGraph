#ifndef KERNEL_TENSOR_H
#define KERNEL_TENSOR_H

#include "layout.h"
#include "hardware/blob.hh"

namespace refactor::kernel {

    using Shape = absl::InlinedVector<dim_t, 4>;
    using Strides = absl::InlinedVector<dim_t, 4>;

    struct Tensor {
        DataType dataType;
        Shape shape;
        LayoutType layout;
        Arc<hardware::Blob> data;

        Tensor(DataType,
               Shape,
               LayoutType,
               Arc<hardware::Blob>) noexcept;
        static Arc<Tensor>
            share(DataType,
                  Shape,
                  LayoutType = LayoutType::Others,
                  Arc<hardware::Blob> = nullptr) noexcept;

        int64_t rank() const;
        size_t elementsSize() const;
        size_t bytesSize() const;
        Strides strides() const;

        void *malloc();
        void free();
    };

    using TensorRefs = absl::InlinedVector<std::reference_wrapper<Tensor const>, 2>;

}// namespace refactor::kernel

#endif// KERNEL_TENSOR_H
