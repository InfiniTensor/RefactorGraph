#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "binary_cudnn.hh"
#include "kernel/kernel.h"
#include "kernel/tensor.h"

namespace refactor::kernel {
    using namespace cudnn;
    using namespace runtime;
    Routine BinaryCudnn::lower() const noexcept {
        struct Descriptors {
            cudnnOpTensorDescriptor_t opDesc;
            cudnnTensorDescriptor_t aDesc, bDesc, cDesc;
            float aAlpha = 1.f;
            float bAlpha = 1.f;
            float beta = 0.f;
            Descriptors() {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&aDesc));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&bDesc));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&cDesc));
                CUDNN_ASSERT(cudnnCreateOpTensorDescriptor(&opDesc));
            }
            ~Descriptors() {
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(aDesc));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(bDesc));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(cDesc));
                CUDNN_ASSERT(cudnnDestroyOpTensorDescriptor(opDesc));
            }
        };
        auto d = std::make_shared<Descriptors>();
        cudnnOpTensorOp_t cudnnOP;
        if (opType == SimpleBinaryType::Add) {
            cudnnOP = CUDNN_OP_TENSOR_ADD;
        } else if (opType == SimpleBinaryType::Sub) {
            cudnnOP = CUDNN_OP_TENSOR_ADD;
            d->bAlpha = -1.f;
        } else if (opType == SimpleBinaryType::Mul) {
            cudnnOP = CUDNN_OP_TENSOR_MUL;
        }

        setCudnnTensor(d->aDesc, dataType, aDims);
        setCudnnTensor(d->bDesc, dataType, bDims);
        setCudnnTensor(d->cDesc, dataType, cDims);
        CUDNN_ASSERT(cudnnSetOpTensorDescriptor(
            d->opDesc, cudnnOP, cudnnDataTypeConvert(dataType), CUDNN_NOT_PROPAGATE_NAN));

        bool swap = aDims != cDims;

        return [d_ = std::move(d), swap](Resources &res, void const **inputs, void **outputs) {
            auto handle = res.fetchOrStore<CudnnContext>()->handle;
            auto const &d = *d_;
            // name inputs and outputs
            auto a = inputs[0],
                 b = inputs[1];
            auto c = outputs[0];
            if (swap) {
                CUDNN_ASSERT(cudnnOpTensor(handle, d.opDesc, &(d.aAlpha),
                                           d.bDesc, b, &(d.bAlpha), d.aDesc, a,
                                           &(d.beta), d.cDesc, c));
            } else {
                CUDNN_ASSERT(cudnnOpTensor(handle, d.opDesc, &(d.aAlpha),
                                           d.aDesc, a, &(d.bAlpha), d.bDesc, b,
                                           &(d.beta), d.cDesc, c));
            }
        };
    }
}// namespace refactor::kernel
