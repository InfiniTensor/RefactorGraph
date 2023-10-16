#ifndef KERNEL_TENSOR_H
#define KERNEL_TENSOR_H

#include "mem_manager/blob.hh"
#include "refactor/common.h"
#include <absl/container/inlined_vector.h>

namespace refactor::kernel {

    using Shape = absl::InlinedVector<uint_lv2, 4>;

    enum class LayoutType : uint8_t {
        NCHW,
        NHWC,
        Others,
    };

    struct Tensor {
        DataType dataType;
        Shape shape;
        LayoutType layout;
        std::shared_ptr<mem_manager::Blob> data;

        Tensor(DataType,
               Shape,
               LayoutType,
               std::shared_ptr<mem_manager::Blob>) noexcept;
        static std::shared_ptr<Tensor>
            share(DataType,
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
