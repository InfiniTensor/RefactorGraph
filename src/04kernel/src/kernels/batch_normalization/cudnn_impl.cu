#include "../cudnn_context.hh"
#include "cudnn_impl.h"
#include "runtime/resource.h"
#include <cudnn.h>

namespace refactor::kernel::cudnn {
    using namespace runtime;
    using Ctx = CudnnContext;

    Operation lower(
        float epsilon,
        common::DataType dataType,
        Shape shape,
        uint32_t valueSize) {
        return [](Resources &res, Addresses inputs, Addresses outputs) {
            auto handle = std::any_cast<cudnnHandle_t>(res.fetchOrStore<CudnnContext>()->handle);
            auto x = inputs[0],
                 scale = inputs[1],
                 bias = inputs[2],
                 mean = inputs[3],
                 var = inputs[4];
            auto y = outputs[0];
        };
    }

}// namespace refactor::kernel::cudnn
