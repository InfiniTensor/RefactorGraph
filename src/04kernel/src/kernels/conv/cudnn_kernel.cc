#include "cudnn_kernel.hh"

#ifdef USE_CUDA
#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#include "../expand/cuda_kernel.hh"
#include "../extra_padding/extra_padding.cuh"
#include "hardware/functions.h"
#endif

namespace refactor::kernel {
    using K = ConvCudnn;

    K::ConvCudnn(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(PoolAttributes const &poolAttributes,
                  Tensor const &x,
                  Tensor const &w,
                  std::optional<std::reference_wrapper<Tensor const>> b,
                  Tensor const &y) -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        std::optional<ExpandInfo> biasExpand = std::nullopt;
        if (b) {
            ASSERT(b->get().shape[0] == y.shape[1], "");
            std::vector<dim_t> input(y.rank(), 1);
            input[1] = y.shape[1];
            biasExpand.emplace(ExpandInfo(
                b->get().dataType,
                slice(input.data(), input.size()),
                slice(y.shape.data(), y.rank())));
        }

        // group is not supported
        if (w.rank() != 4 || poolAttributes.rank() != 2) {
            return nullptr;
        }
        auto d = poolAttributes.dilations(),
             p = poolAttributes.pads(),
             s = poolAttributes.strides();
        return std::make_unique<K>(decltype(info){
            x.dataType,
            {
                static_cast<int>(x.shape[0]),
                static_cast<int>(x.shape[1]),
                static_cast<int>(x.shape[2]),
                static_cast<int>(x.shape[3]),
            },
            {
                static_cast<int>(w.shape[0]),
                static_cast<int>(w.shape[1]),
                static_cast<int>(w.shape[2]),
                static_cast<int>(w.shape[3]),
            },
            {
                static_cast<int>(y.shape[0]),
                static_cast<int>(y.shape[1]),
                static_cast<int>(y.shape[2]),
                static_cast<int>(y.shape[3]),
            },
            {d[0], d[1]},
            {p[0], p[1], p[2], p[3]},
            {s[0], s[1]},
            std::move(biasExpand),
        });
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing conv using CUDNN";
    }

#ifdef USE_CUDA

    auto ConvCudnn::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cudnn;
        using namespace runtime;

        // RAII for closure
        struct Descriptors {
            cudnnTensorDescriptor_t x, y;
            cudnnFilterDescriptor_t w;
            cudnnConvolutionDescriptor_t conv;
            cudnnConvolutionFwdAlgo_t algo;
            std::optional<ExtraPadding> extraPadding;
            std::optional<Routine> biasExpand;
            bool f32;

            Descriptors(decltype(f32) f32_)
                : extraPadding(std::nullopt),
                  biasExpand(std::nullopt),
                  f32(f32_) {
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
        auto d = std::make_shared<Descriptors>(info.dt != DataType::F64);
        d->extraPadding = ExtraPadding::build(info.dt, info.xShape, info.pad);
        if (info.biasExpand) {
            d->biasExpand = ExpandCuda(*info.biasExpand).lower(res).routine;
        }
        int xs[]{
            info.xShape[0],
            info.xShape[1],
            info.xShape[2] + std::abs(info.pad[0] - info.pad[2]),
            info.xShape[3] + std::abs(info.pad[1] - info.pad[3]),
        };

        setCudnnTensor(d->x, info.dt, slice(xs, 4));
        setCudnnTensor(d->y, info.dt, slice(info.yShape, 4));
        auto ws = info.wShape;
        CUDNN_ASSERT(cudnnSetFilter4dDescriptor(
            d->w, cudnnDataTypeConvert(info.dt), CUDNN_TENSOR_NCHW,
            ws[0], ws[1], ws[2], ws[3]));
        auto pp = info.pad;
        auto ss = info.stride;
        auto dd = info.dilation;
        CUDNN_ASSERT(cudnnSetConvolution2dDescriptor(
            d->conv,
            std::min(pp[0], pp[2]), std::min(pp[1], pp[3]),
            ss[0], ss[1],
            dd[0], dd[1],
            CUDNN_CROSS_CORRELATION,
            cudnnDataTypeConvert(d->f32 ? DataType::F32 : DataType::F64)));

        if (auto group = xs[1] / ws[1]; group > 1) {
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
            if (auto f = d->extraPadding; f) {
                x = (*f)(x, reinterpret_cast<uint8_t *>(workspace) + workspaceSize);
            }
            if (auto f = d->biasExpand; f) {
                (*f)(res, workspace, inputs + 2, outputs);
            }
            // build alpha/beta for double
            // build alpha/beta for double
            auto a = d->f32 ? factor<fp32_t>(1) : factor<fp64_t>(1),
                 b = d->f32
                         ? factor<fp32_t>(d->biasExpand ? 1 : 0)
                         : factor<fp64_t>(d->biasExpand ? 1 : 0);
            CUDNN_ASSERT(cudnnConvolutionForward(
                res.fetchOrStore<CudnnContext>()->handle,
                &a,
                d->x, x,
                d->w, w,
                d->conv, d->algo,
                workspace, workspaceSize,
                &b,
                d->y, outputs[0]));
        };
        return {
            std::move(routine),
            workspaceSize + (d->extraPadding ? d->extraPadding->workspace() : 0),
        };
    }

#endif

}// namespace refactor::kernel
