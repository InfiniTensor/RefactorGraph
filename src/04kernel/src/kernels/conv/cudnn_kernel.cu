#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "../expand/cuda_kernel.hh"
#include "cudnn_kernel.hh"
#include "hardware/functions.h"
#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>

namespace refactor::kernel {
    using namespace cudnn;
    using namespace runtime;

    struct ExtraPadding {
        DataType dt;
        int nc, sohw, sow, h, w, padH, padW;

        static std::optional<ExtraPadding> build(DataType dt, int const *shape, int const *pads) {
            if (pads[0] == pads[2] && pads[1] == pads[3]) {
                return std::nullopt;
            }
            int padH = pads[0] - pads[2], padW = pads[1] - pads[3];
            return ExtraPadding{
                dt,
                shape[0] * shape[1],
                (shape[2] + std::abs(padH)) * (shape[3] + std::abs(padW)),
                shape[3] + std::abs(padW),
                shape[2],
                shape[3],
                padH,
                padW};
        }

        size_t workspace() const {
            return nc * sohw * dt.size();
        }
    };

    template<class T>
    struct ExtraPaddingFunctor {
        ExtraPadding info;
        void const *src;

        __device__ T operator()(size_t i) const noexcept {
            auto h = i / info.sow,
                 w = i % info.sow;
            if (0 < info.padH) {
                if (h < info.padH) {
                    return 0;
                }
                h -= info.padH;
            } else if (h >= info.h) {
                return 0;
            }
            if (0 < info.padW) {
                if (w < info.padW) {
                    return 0;
                }
                w -= info.padW;
            } else if (w >= info.w) {
                return 0;
            }
            return reinterpret_cast<T const *>(src)[i / info.sohw * info.h * info.w + h * info.w + w];
        }
    };

    auto ConvCudnn::lower(Resources &res) const -> RoutineWorkspace {
        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t x, y;
            cudnnFilterDescriptor_t w;
            cudnnConvolutionDescriptor_t conv;
            cudnnConvolutionFwdAlgo_t algo;
            std::optional<ExtraPadding> extraPadding;
            std::optional<Routine> biasExpand;
            bool f64;

            Descriptors(bool f64_) : extraPadding(std::nullopt),
                                     biasExpand(std::nullopt),
                                     f64(f64_) {
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
        d->extraPadding = ExtraPadding::build(info.dt, info.xShape, info.pad);
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
        CUDNN_ASSERT(cudnnSetConvolution2dDescriptor(
            d->conv,
            std::min(pp[0], pp[2]), std::min(pp[1], pp[3]),
            ss[0], ss[1],
            dd[0], dd[1],
            CUDNN_CROSS_CORRELATION,
            cudnnDataType));

        if (auto group = info.xShape[1] / ws[1]; group > 1) {
            CUDNN_ASSERT(cudnnSetConvolutionGroupCount(d->conv, group));
        }

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
            // d->algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        }
        size_t workspaceSize;
        {
            CUDNN_ASSERT(cudnnGetConvolutionForwardWorkspaceSize(
                handle,
                d->x, d->w, d->conv, d->y,
                d->algo,
                &workspaceSize));
        }
        if (d->extraPadding) {
            workspaceSize = hardware::alignBytes(workspaceSize, 256);
        }

        // nvcc at c++11 doesn't support real move capture
        auto routine = [d, workspaceSize](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            void const *x = inputs[0], *w = inputs[1];
            if (d->extraPadding) {
                auto extra = reinterpret_cast<uint8_t *>(workspace) + workspaceSize;
                thrust::tabulate(thrust::device,
                                 extra, extra + d->extraPadding->workspace(),
                                 ExtraPaddingFunctor<float>{*d->extraPadding, x});
                x = extra;
            }
            if (d->biasExpand) { (*(d->biasExpand))(res, workspace, inputs + 2, outputs); }
            // build alpha/beta for double
            union {
                float f32[2];
                double f64[2];
            };
            void *alpha, *beta;
            if (d->f64) {
                f64[0] = 1;
                f64[1] = d->biasExpand ? 1 : 0;
                alpha = f64;
                beta = f64 + 1;
            } else {
                f32[0] = 1;
                f32[1] = d->biasExpand ? 1 : 0;
                alpha = f32;
                beta = f32 + 1;
            }
            CUDNN_ASSERT(cudnnConvolutionForward(
                res.fetchOrStore<CudnnContext>()->handle,
                alpha,
                d->x, x,
                d->w, w,
                d->conv, d->algo,
                workspace, workspaceSize,
                beta,
                d->y, outputs[0]));
        };
        return {std::move(routine), d->extraPadding ? workspaceSize + d->extraPadding->workspace() : workspaceSize};
    }

}// namespace refactor::kernel
