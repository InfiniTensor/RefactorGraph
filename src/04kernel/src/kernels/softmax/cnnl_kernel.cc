#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#endif

namespace refactor::kernel {
    using K = SoftmaxCnnl;

    K::SoftmaxCnnl(cnnl::SoftmaxAlgo algo_, DataType type_,
                   int pre_, int mid_, int post_) noexcept
        : Kernel(), algo(algo_), dataType(type_),
          pre(pre_), mid(mid_), post(post_) {}

    auto K::build(cnnl::SoftmaxAlgo algo, SoftmaxInfo info) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif

        return std::make_unique<K>(algo, info.type, info.pre, info.mid, info.post);
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing softmax forward with CNNL";
    }

#ifdef USE_BANG

    auto SoftmaxCnnl::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;

        // RAII for closure
        struct Descriptors {
            cnnlTensorDescriptor_t t;
            cnnlSoftmaxAlgorithm_t algo;
            bool f32;

            Descriptors(decltype(algo) algo_, decltype(f32) f32_)
                : algo(algo_), f32(f32_) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&t));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(t));
            }
            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };

        auto d = std::make_shared<Descriptors>(
            static_cast<cnnlSoftmaxAlgorithm_t>(algo),
            dataType != DataType::F64);
        int dims[]{pre, mid, post};
        // cnnlSoftmaxMode_t mode = (pre == 1)  ? CNNL_SOFTMAX_MODE_HIGH_DIMENSION
        //                          : (post == 1) ? CNNL_SOFTMAX_MODE_LOW_DIMENSION
        //                                       : CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;
        // FIXME(bolun): CNNL Softmax mode
        cnnlSoftmaxMode_t mode = CNNL_SOFTMAX_MODE_MEDIUM_DIMENSION;

        // cnnlSoftmaxForward_v2 is applied to a 3D input tensor only
        CNNL_ASSERT(cnnlSetTensorDescriptor(d->t, CNNL_LAYOUT_ARRAY, cnnlDataTypeConvert(dataType), 3, dims));

        res.fetchOrStore<CnnlContext>();
        return [d = std::move(d), mode](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            // build alpha/beta for double
            auto a = d->f32 ? factor<fp32_t>(1) : factor<fp64_t>(1),
                 b = d->f32 ? factor<fp32_t>(0) : factor<fp64_t>(0);
            CNNL_ASSERT(cnnlSoftmaxForward_v2(
                res.fetchOrStore<CnnlContext>()->handle,
                d->algo,
                mode,
                CNNL_COMPUTATION_ULTRAHIGH_PRECISION,
                &a, d->t, inputs[0],
                &b, d->t, outputs[0]));
        };
    }

#endif

}// namespace refactor::kernel
