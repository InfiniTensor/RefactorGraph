#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "../expand/cuda_kernel.hh"
#include "cudnn_kernel.hh"

namespace refactor::kernel {
    using namespace cudnn;
    using namespace runtime;

    auto ConvCudnn::lower(Resources &res) const -> RoutineWorkspace {
        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t x, y;
            cudnnFilterDescriptor_t w;
            cudnnConvolutionDescriptor_t conv;
            cudnnConvolutionFwdAlgo_t algo;
            std::optional<Routine> biasExpand;
            bool f64;

            Descriptors(bool f64_) : biasExpand(std::nullopt), f64(f64_) {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&x));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&y));
                CUDNN_ASSERT(cudnnCreateFilterDescriptor(&w));
                CUDNN_ASSERT(cudnnCreateConvolutionDescriptor(&conv));
            }
            ~Descriptors() noexcept(false) {
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(x));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(y));
                CUDNN_ASSERT(cudnnDestroyFilterDescriptor(w));
                CUDNN_ASSERT(cudnnDestroyConvolutionDescriptor(conv));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>(info.dt == DataType::F64);
        if (info.biasExpand) {
            d->biasExpand = ExpandCuda(*info.biasExpand).lower(res).routine;
        }

        auto cudnnDataType = cudnnDataTypeConvert(info.dt);
        setCudnnTensor(d->x, info.dt, slice(info.xShape, 4));
        setCudnnTensor(d->y, info.dt, slice(info.yShape, 4));
        auto ws = info.wShape;
        CUDNN_ASSERT(cudnnSetFilter4dDescriptor(d->w, cudnnDataType, CUDNN_TENSOR_NCHW, ws[0], ws[1], ws[2], ws[3]));
        auto pp = info.pad;
        auto ss = info.stride;
        auto dd = info.dilation;
        CUDNN_ASSERT(cudnnSetConvolution2dDescriptor(d->conv, pp[0], pp[1], ss[0], ss[1], dd[0], dd[1], CUDNN_CROSS_CORRELATION, cudnnDataType));

        auto handle = res.fetchOrStore<CudnnContext>()->handle;
        {
            int returnedAlgoCount;
            cudnnConvolutionFwdAlgoPerf_t perfResults;
            CUDNN_ASSERT(cudnnFindConvolutionForwardAlgorithm(
                handle,
                d->x, d->w, d->conv, d->y,
                1, &returnedAlgoCount, &perfResults));
            ASSERT(returnedAlgoCount == 1, "returnedAlgoCount != 1");
            d->algo = perfResults.algo;
            // for high accuracy, use this algo only
            d->algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        }
        size_t workspaceSize;
        {
            CUDNN_ASSERT(cudnnGetConvolutionForwardWorkspaceSize(
                handle,
                d->x, d->w, d->conv, d->y,
                d->algo,
                &workspaceSize));
        }
        // nvcc at c++11 doesn't support real move capture
        auto routine = [d_ = std::move(d),
                        workspaceSize](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            auto const &d = *d_;
            if (d.biasExpand) { (*(d.biasExpand))(res, workspace, inputs + 2, outputs); }
            // fetch cudnn handle from resources
            auto handle = res.fetchOrStore<CudnnContext>()->handle;
            // build alpha/beta for double
            union {
                float f32[2];
                double f64[2];
            };
            void *alpha, *beta;
            if (d.f64) {
                f64[0] = 1;
                f64[1] = d.biasExpand ? 1 : 0;
                alpha = f64;
                beta = f64 + 1;
            } else {
                f32[0] = 1;
                f32[1] = d.biasExpand ? 1 : 0;
                alpha = f32;
                beta = f32 + 1;
            }
            CUDNN_ASSERT(cudnnConvolutionForward(
                handle,
                alpha,
                d.x, inputs[0],
                d.w, inputs[1],
                d.conv, d.algo,
                workspace, workspaceSize,
                beta,
                d.y, outputs[0]));
        };
        return {std::move(routine), workspaceSize};
    }

}// namespace refactor::kernel
