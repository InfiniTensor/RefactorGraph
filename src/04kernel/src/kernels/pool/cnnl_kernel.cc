#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#endif

namespace refactor::kernel {
    using K = PoolCnnl;

    K::PoolCnnl(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(PoolType poolType,
                  bool ceil,
                  KernelShape const &kernelShape,
                  PoolAttributes const &poolAttributes,
                  Tensor const &x,
                  Tensor const &y) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif

        // TODO check data type
        auto p = poolAttributes.pads(),
             d = poolAttributes.dilations(),
             s = poolAttributes.strides();
        if (x.rank() != 4 ||
            poolType == PoolType::Lp ||
            d[0] != 1 || d[1] != 1) {
            return nullptr;
        }
        return std::make_unique<K>(decltype(info){
            poolType,
            x.dataType,
            {
                static_cast<int>(x.shape[0]),
                static_cast<int>(x.shape[1]),
                static_cast<int>(x.shape[2]),
                static_cast<int>(x.shape[3]),
            },
            {
                static_cast<int>(y.shape[0]),
                static_cast<int>(y.shape[1]),
                static_cast<int>(y.shape[2]),
                static_cast<int>(y.shape[3]),
            },
            {
                static_cast<int>(kernelShape[0]),
                static_cast<int>(kernelShape[1]),
            },
            {p[0], p[1], p[2], p[3]},
            {s[0], s[1]},
            {d[0], d[1]},
            ceil
        });
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing pool using CNNL";
    }

#ifdef USE_BANG

    auto PoolCnnl::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;
        using Ty = PoolType;

        // RAII for closure
        struct Descriptors {
            cnnlTensorDescriptor_t x, y;
            cnnlPoolingDescriptor_t pooling;
            bool f32;

            Descriptors(decltype(f32) f32_) : f32(f32_) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&x));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&y));
                CNNL_ASSERT(cnnlCreatePoolingDescriptor(&pooling));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(x));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(y));
                CNNL_ASSERT(cnnlDestroyPoolingDescriptor(pooling));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>(info.dt != DataType::F64);
        int const
            xs[]{
                info.xShape[0],
                info.xShape[1],
                info.xShape[2] + std::abs(info.pads[0] - info.pads[2]),
                info.xShape[3] + std::abs(info.pads[1] - info.pads[3]),
            },
            *ys = info.yShape;
        setCnnlTensor(d->x, info.dt, slice(xs, 4));
        setCnnlTensor(d->y, info.dt, slice(ys, 4));

        // clang-format off
        auto mode = info.poolType == Ty::Average ? CNNL_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
                  : info.poolType == Ty::Max     ? CNNL_POOLING_MAX
                                                 : UNREACHABLEX(cnnlPoolingMode_t, "");
        // clang-format on
        auto pp = info.pads;
        auto ss = info.strides;
        auto kk = info.kernelShape;
        auto dd = info.dilations;
        CNNL_ASSERT(cnnlSetPooling2dDescriptor_v2(
            d->pooling, mode, CNNL_NOT_PROPAGATE_NAN,
            kk[0], kk[1], pp[0], pp[2], pp[1], pp[3],
            ss[0], ss[1], dd[0], dd[1], ceil));

        auto handle = res.fetchOrStore<CnnlContext>()->handle;
        size_t extraInputSize, workspaceSize;
        CNNL_ASSERT(cnnlGetPoolingWorkspaceSize(handle, mode, ys[3], ys[2], &workspaceSize));
        CNNL_ASSERT(cnnlGetPoolingExtraInputSize(handle, mode, ys[3], ys[2], &extraInputSize));

        res.fetchOrStore<CnnlContext>();
        auto routine = [d, workspaceSize,
                        extraInputSize](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            auto handle = res.fetchOrStore<CnnlContext>()->handle;
           
            void *extraInputDev = workspace;
            void *poolWorkSpace = reinterpret_cast<uint8_t *>(workspace) + extraInputSize;

            void *extraInputHost = malloc(extraInputSize);
            CNNL_ASSERT(cnnlInitPoolingExtraInput(handle, d->pooling, d->x, d->y, extraInputHost));
            BANG_ASSERT(cnrtMemcpy(extraInputDev, extraInputHost, extraInputSize, CNRT_MEM_TRANS_DIR_HOST2DEV));

            // build alpha/beta for double
            auto a = d->f32 ? factor<fp32_t>(1) : factor<fp64_t>(1),
                 b = d->f32 ? factor<fp32_t>(0) : factor<fp64_t>(0);
            CNNL_ASSERT(cnnlPoolingForward_v2(
                handle, d->pooling,
                &a, d->x, inputs[0],
                &b, extraInputDev, d->y, outputs[0],
                poolWorkSpace, workspaceSize));

            res.fetchOrStore<CnnlContext>()->queueSync();

            free(extraInputHost);
        };
        return {std::move(routine), workspaceSize + extraInputSize};
    }
#endif

}// namespace refactor::kernel
