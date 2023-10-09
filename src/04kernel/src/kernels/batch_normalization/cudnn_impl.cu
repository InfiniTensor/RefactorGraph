#include "../cudnn_context.hh"
#include "../cudnn_error_handler.h"
#include "cudnn_impl.h"
#include "runtime/resource.h"
#include <cudnn.h>
#include <utility>

namespace refactor::kernel::cudnn {
    using namespace runtime;
    using Ctx = CudnnContext;
    using DT = common::DataType;

    Operation lower(
        float epsilon,
        std::array<DT, 3> dts,
        Shape shape,
        uint32_t paramSize) {

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
        dimAp[1] = paramSize;
        strideAp[0] = paramSize;

        struct Descriptors {
            cudnnTensorDescriptor_t x, param;

            Descriptors() : x(nullptr), param(nullptr) {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&x),
                             "cudnnCreateTensorDescriptor failed");
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&param),
                             "cudnnCreateTensorDescriptor failed");
            }
            ~Descriptors() {
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(x),
                             "cudnnDestroyTensorDescriptor failed");
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(param),
                             "cudnnDestroyTensorDescriptor failed");
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>();

        cudnnDataType_t dts_[3];
        for (size_t i = 0; i < 3; ++i) {
            switch (dts[i]) {
                case DT::F32:
                    dts_[i] = CUDNN_DATA_FLOAT;
                    break;
                case DT::F64:
                    dts_[i] = CUDNN_DATA_DOUBLE;
                    break;
                case DT::FP16:
                    dts_[i] = CUDNN_DATA_HALF;
                    break;
                case DT::BF16:
                    dts_[i] = CUDNN_DATA_BFLOAT16;
                    break;
                default:
                    UNREACHABLE();
            }
        }

        CUDNN_ASSERT(cudnnSetTensorNdDescriptor(d->x, dts_[0], rank, dimAx, strideAx),
                     "cudnnSetTensorNdDescriptor failed");
        CUDNN_ASSERT(cudnnSetTensorNdDescriptor(d->param, dts_[1], rank, dimAp, strideAp),
                     "cudnnSetTensorNdDescriptor failed");

        // nvcc at c++11 doesn't support real move capture
        return [d = std::move(d), epsilon](Resources &res, Addresses inputs, Addresses outputs) {
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
                d->x, x, d->x, y, d->param, scale, bias, mean, var, epsilon);
            if (stat != CUDNN_STATUS_SUCCESS) {
                RUNTIME_ERROR("cudnnBatchNormalizationForwardInference failed");
            }
        };
    }

}// namespace refactor::kernel::cudnn
