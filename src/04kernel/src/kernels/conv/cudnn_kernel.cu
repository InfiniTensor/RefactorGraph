#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "cudnn_kernel.hh"
#include "runtime/mem_manager.hh"

namespace refactor::kernel {
    using namespace cudnn;
    using namespace runtime;

    auto ConvCudnn::lower(Resources &res) const noexcept -> Routine {
        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t x, y;
            cudnnFilterDescriptor_t w;
            cudnnConvolutionDescriptor_t conv;
            cudnnConvolutionFwdAlgo_t algo;
            size_t workspaceSize;

            Descriptors() : workspaceSize(0) {
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

        auto cudnnDataType = cudnnDataTypeConvert(info.dt);
        auto xs = info.xShape, ys = info.yShape, ws = info.wShape;
        CUDNN_ASSERT(cudnnSetTensor4dDescriptor(d->x, CUDNN_TENSOR_NCHW, cudnnDataType, xs[0], xs[1], xs[2], xs[3]));
        CUDNN_ASSERT(cudnnSetTensor4dDescriptor(d->y, CUDNN_TENSOR_NCHW, cudnnDataType, ys[0], ys[1], ys[2], ys[3]));
        CUDNN_ASSERT(cudnnSetFilter4dDescriptor(d->w, cudnnDataType, CUDNN_TENSOR_NCHW, ws[0], ws[1], ws[2], ws[3]));
        auto pp = info.pad;
        auto ss = info.stride;
        auto dd = info.dilation;
        CUDNN_ASSERT(cudnnSetConvolution2dDescriptor(d->conv, pp[0], pp[1], ss[0], ss[1], dd[0], dd[1], CUDNN_CROSS_CORRELATION, cudnnDataType));

        auto handle = res.fetchOrStore<CudnnContext>()->handle;
        int returnedAlgoCount;
        cudnnConvolutionFwdAlgoPerf_t perfResults;
        CUDNN_ASSERT(cudnnFindConvolutionForwardAlgorithm(
            handle,
            d->x, d->w, d->conv, d->y,
            1, &returnedAlgoCount, &perfResults));
        ASSERT(returnedAlgoCount == 1, "returnedAlgoCount != 1");
        d->algo = perfResults.algo;
        CUDNN_ASSERT(cudnnGetConvolutionForwardWorkspaceSize(
            handle,
            d->x, d->w, d->conv, d->y,
            perfResults.algo,
            &d->workspaceSize));
        // nvcc at c++11 doesn't support real move capture
        return [d_ = std::move(d)](Resources &res, void const **inputs, void **outputs) {
            using mem_manager::ForeignBlob;
            auto const &d = *d_;
            // fetch cudnn handle from resources
            auto handle = res.fetchOrStore<CudnnContext>()->handle;
            auto workspace = ForeignBlob::share(res.fetch<MemManager>()->manager, d.workspaceSize);
            // TODO? build alpha/beta for double
            float alpha = 1, beta = 0;
            CUDNN_ASSERT(cudnnConvolutionForward(
                handle,
                &alpha,
                d.x, inputs[0],
                d.w, inputs[1],
                d.conv, d.algo,
                *workspace, d.workspaceSize,
                &beta,
                d.y, outputs[0]));
        };
    }

}// namespace refactor::kernel
