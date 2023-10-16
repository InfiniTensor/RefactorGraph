#include "kernel/tensor.h"
#include <numeric>

namespace refactor::kernel {

    Tensor::Tensor(DataType dataType_,
                   Shape shape_,
                   LayoutType layout_,
                   std::shared_ptr<mem_manager::Blob> data_) noexcept
        : dataType(dataType_),
          shape(std::move(shape_)),
          layout(layout_),
          data(std::move(data_)) {}

    std::shared_ptr<Tensor>
    Tensor::share(DataType dataType,
                  Shape shape,
                  LayoutType layout,
                  std::shared_ptr<mem_manager::Blob> data) noexcept {
        return std::make_shared<Tensor>(dataType, std::move(shape), layout, std::move(data));
    }

    int64_t Tensor::rank() const { return shape.size(); }
    size_t Tensor::elementsSize() const { return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()); }
    size_t Tensor::bytesSize() const { return dataType.size() * elementsSize(); }

}// namespace refactor::kernel
