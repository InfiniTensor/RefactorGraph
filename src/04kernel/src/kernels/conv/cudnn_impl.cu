#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "cudnn_kernel.hh"
#include "runtime/mem_manager.hh"

namespace refactor::kernel {
    using namespace cudnn;
    using namespace runtime;

    auto ConvCudnn::lower() const noexcept -> Routine {
        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t x, y;
            cudnnFilterDescriptor_t w;
            cudnnConvolutionDescriptor_t conv;
            cudnnConvolutionFwdAlgo_t algo;

            Descriptors() {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&x));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&y));
                CUDNN_ASSERT(cudnnCreateFilterDescriptor(&w));
                CUDNN_ASSERT(cudnnCreateConvolutionDescriptor(&conv));
            }
            ~Descriptors() {
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(x));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(y));
                CUDNN_ASSERT(cudnnDestroyFilterDescriptor(w));
                CUDNN_ASSERT(cudnnDestroyConvolutionDescriptor(conv));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>();

        d->algo = static_cast<cudnnConvolutionFwdAlgo_t>(info.algo);
        auto cudnnDataType = cudnnDataTypeConvert(info.dt);
        auto xs = info.xShape, ys = info.yShape, ws = info.wShape;
        CUDNN_ASSERT(cudnnSetTensor4dDescriptor(d->x, CUDNN_TENSOR_NCHW, cudnnDataType, xs[0], xs[1], xs[2], xs[3]));
        CUDNN_ASSERT(cudnnSetTensor4dDescriptor(d->y, CUDNN_TENSOR_NCHW, cudnnDataType, ys[0], ys[1], ys[2], ys[3]));
        CUDNN_ASSERT(cudnnSetFilter4dDescriptor(d->w, cudnnDataType, CUDNN_TENSOR_NCHW, ws[0], ws[1], ws[2], ws[3]));
        auto pp = info.pad;
        auto ss = info.stride;
        auto dd = info.dilation;
        CUDNN_ASSERT(cudnnSetConvolution2dDescriptor(d->conv, pp[0], pp[1], ss[0], ss[1], dd[0], dd[1], CUDNN_CROSS_CORRELATION, cudnnDataType));

        // nvcc at c++11 doesn't support real move capture
        return [d_ = std::move(d)](Resources &res, void const **inputs, void **outputs) {
            // fetch cudnn handle from resources
            auto handle = res.fetchOrStore<CudnnContext>()->handle;
            constexpr static auto workspaceSize = 1ul << 30;
            auto workspace = mem_manager::ForeignBlob::share(res.fetch<MemManager>()->manager, workspaceSize);
            auto const &d = *d_;
            // name inputs and outputs
            auto x = inputs[0],
                 w = inputs[1];
            auto y = outputs[0];
            // TODO? build alpha/beta for double
            float alpha = 1, beta = 0;
            CUDNN_ASSERT(cudnnConvolutionForward(
                handle,
                &alpha,
                d.x, x,
                d.w, w,
                d.conv, d.algo,
                *workspace, workspaceSize,
                &beta,
                d.y, y));
        };
    }

}// namespace refactor::kernel
