#include "computation/tensor.h"

namespace refactor::computation {
    Tensor::Tensor(common::DataType dataType_,
                   Shape shape_,
                   LayoutType layout_,
                   std::shared_ptr<mem_manager::Blob> data_)
        : dataType(dataType_),
          shape(std::move(shape_)),
          layout(layout_),
          data(std::move(data_)) {}

    std::shared_ptr<Tensor>
    Tensor::share(common::DataType dataType,
                  Shape shape,
                  LayoutType layout,
                  std::shared_ptr<mem_manager::Blob> data) {
        return std::make_shared<Tensor>(dataType, std::move(shape), layout, std::move(data));
    }

}// namespace refactor::computation
