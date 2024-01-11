#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#include <cnnl.h>
#endif

namespace refactor::kernel {
    using K = BatchNormalizationCnnl;
    using DT = DataType;

    K::BatchNormalizationCnnl(decltype(info) info_) noexcept
        : info(info_) {}

    auto K::build(float epsilon, TensorRefs inputs) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif

        auto const &x = inputs[0].get();
        auto const &scale = inputs[1].get();
        auto const &mean = inputs[3].get();

        if (x.rank() != 4) {
            return nullptr;
        }

        // see "Supported Configurations for `cnnlBatchNormalizationForwardInference`"
        if (scale.dataType != mean.dataType) {
            return nullptr;
        }
        if (x.dataType == DT::F64) {
            if (scale.dataType != DT::F64) {
                return nullptr;
            }
        } else {
            if (scale.dataType != DT::F32) {
                return nullptr;
            }
        }
        return std::make_unique<K>(decltype(info){
            epsilon,
            x.dataType,
            scale.dataType,
            x.layout,
            {
                static_cast<int>(x.shape[0]),
                static_cast<int>(x.shape[1]),
                static_cast<int>(x.shape[2]),
                static_cast<int>(x.shape[3]),
            }});
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing batch normalization for non-training-mode using CNNL";
    }

#ifdef USE_BANG

    auto K::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;
        using DT = DataType;

        // RAII for closure
        struct Descriptors {
            cnnlTensorDescriptor_t inDesc, inDescTrans, p;
            cnnlTransposeDescriptor_t NCHW2NHWC, NHWC2NCHW;
            bool f32;

            explicit Descriptors(decltype(f32) f32_)
                : inDesc(nullptr), inDescTrans(nullptr), p(nullptr),
                  NCHW2NHWC(nullptr), NHWC2NCHW(nullptr), f32(f32_) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&inDesc));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&inDescTrans));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&p));
                CNNL_ASSERT(cnnlCreateTransposeDescriptor(&NCHW2NHWC));
                CNNL_ASSERT(cnnlCreateTransposeDescriptor(&NHWC2NCHW));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(inDesc));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(inDescTrans));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(p));
                CNNL_ASSERT(cnnlDestroyTransposeDescriptor(NCHW2NHWC));
                CNNL_ASSERT(cnnlDestroyTransposeDescriptor(NHWC2NCHW));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>(info.dtX != DT::F64);
        int dimNCHW[4] = {info.dimAx[0], info.dimAx[1], info.dimAx[2], info.dimAx[3]};
        int dimNHWC[4] = {info.dimAx[0], info.dimAx[2], info.dimAx[3], info.dimAx[1]};
        int dimParam[]{info.dimAx[1]};
        setCnnlTensor(d->inDesc, info.dtX, slice(dimNCHW, 4));
        CNNL_ASSERT(cnnlSetTensorDescriptor(d->inDescTrans, CNNL_LAYOUT_NHWC, cnnlDataTypeConvert(info.dtX), 4, dimNHWC));
        CNNL_ASSERT(cnnlSetTensorDescriptor(d->p, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(info.dtP), 1, dimParam));
        int permute[4] = {0, 2, 3, 1};
        int permuteOut[4] = {0, 3, 1, 2};
        CNNL_ASSERT(cnnlSetTransposeDescriptor(d->NCHW2NHWC, 4, permute));
        CNNL_ASSERT(cnnlSetTransposeDescriptor(d->NHWC2NCHW, 4, permuteOut));

        auto handle = res.fetchOrStore<CnnlContext>()->handle;
        auto xTransSize = cnnlGetTensorElementNum(d->inDescTrans) * info.dtX.size();
        size_t workspaceSize;
        CNNL_ASSERT(cnnlGetTransposeWorkspaceSize(handle, d->inDesc, d->NCHW2NHWC, &workspaceSize));
        size_t totalWorkspaceSize = xTransSize * 2 + workspaceSize;

        res.fetchOrStore<CnnlContext>();
        auto routine = [d = std::move(d),
                        epsilon = info.epsilon,
                        xTransSize, workspaceSize](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            // fetch cnnl handle from resources
            auto handle = res.fetchOrStore<CnnlContext>()->handle;

            // name inputs and outputs
            auto x = inputs[0],
                 scale = inputs[1],
                 bias = inputs[2],
                 mean = inputs[3],
                 var = inputs[4];
            auto y = outputs[0];

            void *xTrans = workspace;
            void *yTrans = xTrans + xTransSize;
            void *cursor = yTrans + xTransSize;

            // transpose NCHW input to NHWC
            CNNL_ASSERT(cnnlTranspose_v2(handle, d->NCHW2NHWC, d->inDesc, x,
                                         d->inDescTrans, xTrans, cursor, workspaceSize));

            // build alpha/beta for double
            auto a = d->f32 ? factor<fp32_t>(1) : factor<fp64_t>(1),
                 b = d->f32 ? factor<fp32_t>(0) : factor<fp64_t>(0);
            CNNL_ASSERT(cnnlBatchNormForwardInference(
                handle, &a, &b,
                d->inDescTrans, xTrans, d->p, scale, bias, mean, var,
                epsilon, d->inDescTrans, yTrans));

            // transpose NHWC intermediates to NCHW
            CNNL_ASSERT(cnnlTranspose_v2(handle, d->NHWC2NCHW, d->inDescTrans, yTrans,
                                         d->inDesc, y, cursor, workspaceSize));

            BANG_ASSERT(cnrtQueueSync(res.fetchOrStore<CnnlContext>()->queue));
        };

        return {std::move(routine), totalWorkspaceSize};
    }

#endif

}// namespace refactor::kernel
