#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#include "../expand/cnnl_kernel.hh"
#include "hardware/functions.h"
#endif

namespace refactor::kernel {
    using K = ConvCnnl;

    K::ConvCnnl(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(PoolAttributes const &poolAttributes,
                  Tensor const &x,
                  Tensor const &w,
                  std::optional<std::reference_wrapper<Tensor const>> b,
                  Tensor const &y) -> KernelBox {
        static const std::unordered_set<decltype(DataType::internal)>
            SET{DataType::FP16, DataType::BF16, DataType::F32, DataType::F64, DataType::I8};
#ifndef USE_BANG
        return nullptr;
#endif

        auto dt = x.dataType;
        if (!SET.contains(dt) || w.dataType != dt || y.dataType != dt) {
            return nullptr;
        }

        std::optional<ExpandInfoCnnl> biasExpand = std::nullopt;
        if (b) {
            ASSERT(b->get().shape[0] == y.shape[1], "");
            std::vector<dim_t> input(y.rank(), 1);
            input[1] = y.shape[1];
            biasExpand.emplace(ExpandInfoCnnl(
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
            dt,
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
        return "Performing conv using CNNL";
    }

#ifdef USE_BANG

    auto ConvCnnl::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;

        // RAII for closure
        struct Descriptors {
            cnnlTensorDescriptor_t x, y, w;
            cnnlTensorDescriptor_t xTrans, yTrans, wTrans;
            cnnlTransposeDescriptor_t NCHW2NHWC, NHWC2NCHW;
            cnnlConvolutionDescriptor_t conv;
            cnnlConvolutionForwardAlgo_t algo;
            // std::optional<ExtraPadding> extraPadding;
            std::optional<Routine> biasExpand;
            bool f32;

            Descriptors(decltype(f32) f32_)
                :// extraPadding(std::nullopt),
                  biasExpand(std::nullopt),
                  f32(f32_) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&x));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&y));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&w));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&xTrans));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&yTrans));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&wTrans));
                CNNL_ASSERT(cnnlCreateTransposeDescriptor(&NCHW2NHWC));
                CNNL_ASSERT(cnnlCreateTransposeDescriptor(&NHWC2NCHW));
                CNNL_ASSERT(cnnlCreateConvolutionDescriptor(&conv));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(x));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(y));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(w));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(xTrans));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(yTrans));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(wTrans));
                CNNL_ASSERT(cnnlDestroyTransposeDescriptor(NCHW2NHWC));
                CNNL_ASSERT(cnnlDestroyTransposeDescriptor(NHWC2NCHW));
                CNNL_ASSERT(cnnlDestroyConvolutionDescriptor(conv));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>(info.dt != DataType::F64);
        // d->extraPadding = ExtraPadding::build(info.dt, info.xShape, info.pad);
        if (info.biasExpand) {
            d->biasExpand = ExpandCnnl(*info.biasExpand).lower(res).routine;
        }
        int xs[]{
            info.xShape[0],
            info.xShape[1],
            info.xShape[2] + std::abs(info.pad[0] - info.pad[2]),
            info.xShape[3] + std::abs(info.pad[1] - info.pad[3]),
        };

        auto NHWC = [](const int shape[]) -> std::vector<int> {
            return {
                shape[0], shape[2], shape[3], shape[1]};
        };

        std::vector<int> xsNHWC = NHWC(xs);
        std::vector<int> wsNHWC = NHWC(info.wShape);
        std::vector<int> ysNHWC = NHWC(info.yShape);

        setCnnlTensor(d->x, info.dt, slice(xs, 4));
        setCnnlTensor(d->y, info.dt, slice(info.yShape, 4));
        setCnnlTensor(d->w, info.dt, slice(info.wShape, 4));
        CNNL_ASSERT(cnnlSetTensorDescriptor(d->xTrans, CNNL_LAYOUT_NHWC, cnnlDataTypeConvert(info.dt), 4, xsNHWC.data()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(d->yTrans, CNNL_LAYOUT_NHWC, cnnlDataTypeConvert(info.dt), 4, ysNHWC.data()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(d->wTrans, CNNL_LAYOUT_NHWC, cnnlDataTypeConvert(info.dt), 4, wsNHWC.data()));
        
        auto xTransSize = cnnlGetTensorElementNum(d->xTrans) * info.dt.size();
        auto yTransSize = cnnlGetTensorElementNum(d->yTrans) * info.dt.size();
        auto wTransSize = cnnlGetTensorElementNum(d->wTrans) * info.dt.size();

        int permuteIn[4] = {0, 2, 3, 1};
        int permuteOut[4] = {0, 3, 1, 2};
        CNNL_ASSERT(cnnlSetTransposeDescriptor(d->NCHW2NHWC, 4, permuteIn));
        CNNL_ASSERT(cnnlSetTransposeDescriptor(d->NHWC2NCHW, 4, permuteOut));

        size_t xWorkspaceSize, yWorkspaceSize, wWorkspaceSize, convWorkspaceSize;
        auto handle = res.fetchOrStore<CnnlContext>()->handle;
        CNNL_ASSERT(cnnlGetTransposeWorkspaceSize(handle, d->x, d->NCHW2NHWC, &xWorkspaceSize));
        CNNL_ASSERT(cnnlGetTransposeWorkspaceSize(handle, d->w, d->NCHW2NHWC, &wWorkspaceSize));
        CNNL_ASSERT(cnnlGetTransposeWorkspaceSize(handle, d->yTrans, d->NHWC2NCHW, &yWorkspaceSize));

        // clang-format off
        auto computation = info.dt == DataType::F64 ? DataType::F64
                         : info.dt == DataType::I8  ? DataType::I32
                         : DataType::F32;
        // clang-format on
        auto group = xs[1] / info.wShape[1];
        CNNL_ASSERT(cnnlSetConvolutionDescriptor(d->conv, 4, info.pad, info.stride, info.dilation, group, cnnlDataTypeConvert(computation)));
        CNNL_ASSERT(cnnlGetConvolutionForwardAlgorithm(
            handle, d->conv, d->xTrans, d->wTrans, d->yTrans,
            CNNL_CONVOLUTION_FWD_FASTEST, &d->algo));

        CNNL_ASSERT(cnnlGetConvolutionForwardWorkspaceSize(
            handle, d->xTrans, d->wTrans, d->yTrans, NULL,
            d->conv, d->algo, &convWorkspaceSize));

        // if (d->extraPadding) {
        //     workspaceSize = hardware::alignBytes(workspaceSize, 256);
        // }

        size_t workspaceSize = xTransSize + yTransSize + wTransSize + std::max({xWorkspaceSize, wWorkspaceSize, yWorkspaceSize, convWorkspaceSize});

        res.fetchOrStore<CnnlContext>();
        auto routine = [d, xTransSize, yTransSize, wTransSize,
                        xWorkspaceSize, wWorkspaceSize,
                        yWorkspaceSize, convWorkspaceSize](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            auto handle = res.fetchOrStore<CnnlContext>()->handle;
            void const *x = inputs[0], *w = inputs[1];
            void *y = outputs[0];
            // if (auto f = d->extraPadding; f) {
            //     x = (*f)(x, reinterpret_cast<uint8_t *>(workspace) + workspaceSize);
            // }
            // if (auto f = d->biasExpand; f) {
            //     (*f)(res, workspace, inputs + 2, outputs);
            // }

            void *xTrans = workspace;
            void *wTrans = xTrans + xTransSize;
            void *yTrans = wTrans + wTransSize;
            void *opWorkspace = yTrans + yTransSize;

            // transpose NCHW input to NHWC
            CNNL_ASSERT(cnnlTranspose_v2(handle, d->NCHW2NHWC, d->x, x,
                                         d->xTrans, xTrans, opWorkspace, xWorkspaceSize));
            CNNL_ASSERT(cnnlTranspose_v2(handle, d->NCHW2NHWC, d->w, w,
                                         d->wTrans, wTrans, opWorkspace, wWorkspaceSize));
            
            // build alpha/beta for double
            auto a = d->f32 ? factor<fp32_t>(1) : factor<fp64_t>(1),
                 b = d->f32
                         ? factor<fp32_t>(d->biasExpand ? 1 : 0)
                         : factor<fp64_t>(d->biasExpand ? 1 : 0);
            CNNL_ASSERT(cnnlConvolutionForward(
                handle,
                d->conv, d->algo, &a,
                d->xTrans, xTrans, d->wTrans, wTrans,
                NULL, NULL, opWorkspace, convWorkspaceSize,
                &b, d->yTrans, yTrans));
            
            // transpose NHWC intermediates to NCHW
            CNNL_ASSERT(cnnlTranspose_v2(handle, d->NHWC2NCHW, d->yTrans, yTrans,
                                         d->y, y, opWorkspace, yWorkspaceSize));
        };
        return {std::move(routine), workspaceSize};
    }

#endif

}// namespace refactor::kernel
