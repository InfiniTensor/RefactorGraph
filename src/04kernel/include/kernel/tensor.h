#ifndef KERNEL_TENSOR_H
#define KERNEL_TENSOR_H

#include "layout.h"
#include "mem_manager/blob.hh"
#include "common.h"
#include <absl/container/inlined_vector.h>

namespace refactor::kernel {

    using Shape = absl::InlinedVector<uint_lv2, 4>;

    struct Tensor {
        DataType dataType;
        Shape shape;
        LayoutType layout;
        Arc<mem_manager::Blob> data;

        Tensor(DataType,
               Shape,
               LayoutType,
               Arc<mem_manager::Blob>) noexcept;
        static Arc<Tensor>
            share(DataType,
                  Shape,
                  LayoutType = LayoutType::Others,
                  Arc<mem_manager::Blob> = nullptr) noexcept;

        int64_t rank() const;
        size_t elementsSize() const;
        size_t bytesSize() const;

        void *malloc();
        void free();
    };

    using TensorRefs = absl::InlinedVector<std::reference_wrapper<Tensor const>, 2>;

}// namespace refactor::kernel

#endif// KERNEL_TENSOR_H
