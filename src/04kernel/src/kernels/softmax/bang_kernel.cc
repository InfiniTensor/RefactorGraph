#include "bang_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#include <kernel/bang/softmax.hh>
#endif

namespace refactor::kernel {
    using K = SoftmaxBang;

    K::SoftmaxBang(SoftmaxInfo info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(SoftmaxInfo info) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif

        return info.type.isFloat()
                   ? std::make_unique<K>(std::move(info))
                   : nullptr;
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing Softmax using BANG";
    }

#ifdef USE_BANG
    template<class T>
    Routine lowerTypedBang(SoftmaxInfo info) {
        using namespace runtime;
        using namespace cnnl;

        return [info](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            auto mlu_src = inputs[0];
            auto mlu_destination = outputs[0];
            int dimsize = info.mid;
            int stride = info.post;
            int frontsize = info.pre;
            int othersize = frontsize * stride;
            int numBlocks = info.pre * info.post;
            int nDim = 4;
            int axis = 1;
            cnrtDim3_t k_dim;
            cnrtFunctionType_t k_type;
            res.fetchOrStore<CnnlContext>();
            cnrtQueue_t queue = res.fetchOrStore<CnnlContext>()->queue;
            k_dim.x = 4;
            k_dim.y = 1;
            k_dim.z = 1;
            k_type = CNRT_FUNC_TYPE_UNION1;

            bang::KernelLaunchParameters params = {k_dim, k_type, queue};
            bang::launchSoftmax<T>(params, mlu_src, mlu_destination, nDim, axis, othersize, frontsize, dimsize, stride);
        };
    }

    auto SoftmaxBang::lower(Resources &res) const noexcept -> RoutineWorkspace {
        switch (info.type.internal) {
            case DataType::F32:
                return lowerTypedBang<float>(info);
            case DataType::F64:
                return lowerTypedBang<double>(info);
            case DataType::FP16:
                return lowerTypedBang<fp16_t>(info);
            case DataType::BF16:
                return lowerTypedBang<bf16_t>(info);
            default:
                UNREACHABLE();
        }
    }
#endif

}// namespace refactor::kernel
