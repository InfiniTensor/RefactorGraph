#include "../cudnn_context.hh"
#include "../cudnn_error_handler.h"
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

        auto rank = shape.size();

        std::vector<int> ints(rank * 4, 1);
        auto dimAx = ints.data();
        auto strideAx = dimAx + rank;
        auto dimAp = strideAx + rank;
        auto strideAp = dimAp + rank;

        auto temp = 1;
        for (size_t i = 0; i < rank; ++i) {
            auto i_ = rank - i - 1;// reverse
            dimAx[i] = shape[i];
            strideAx[i_] = temp;
            temp *= shape[i];
        }
        dimAp[1] = valueSize;
        strideAp[0] = valueSize;

        cudnnTensorDescriptor_t xDesc;
        CUDNN_ASSERT(cudnnCreateTensorDescriptor(&xDesc),
                     "cudnnCreateTensorDescriptor failed");
        CUDNN_ASSERT(cudnnSetTensorNdDescriptor(xDesc, CUDNN_DATA_FLOAT, rank, dimAx, strideAx),
                     "cudnnSetTensorNdDescriptor failed");

        cudnnTensorDescriptor_t paraDesc;
        CUDNN_ASSERT(cudnnCreateTensorDescriptor(&paraDesc),
                     "cudnnCreateTensorDescriptor failed");
        CUDNN_ASSERT(cudnnSetTensorNdDescriptor(paraDesc, CUDNN_DATA_FLOAT, rank, dimAp, strideAp),
                     "cudnnSetTensorNdDescriptor failed");

        return [xDesc, paraDesc, epsilon](Resources &res, Addresses inputs, Addresses outputs) {
            auto handle = std::any_cast<cudnnHandle_t>(res.fetchOrStore<CudnnContext>()->handle);
            auto x = inputs[0],
                 scale = inputs[1],
                 bias = inputs[2],
                 mean = inputs[3],
                 var = inputs[4];
            auto y = outputs[0];

            float alpha = 1, beta = 0;
            // This mode is intended for use after convolutional layers
            auto stat = cudnnBatchNormalizationForwardInference(
                handle, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta,
                xDesc, x, xDesc, y, paraDesc, scale, bias, mean, var, epsilon);
            if (stat != CUDNN_STATUS_SUCCESS) {
                RUNTIME_ERROR("cudnnBatchNormalizationForwardInference failed");
            }
        };
    }

}// namespace refactor::kernel::cudnn
