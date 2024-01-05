#include "kernel/collectors/simple_unary.h"
#include "../kernels/simple_unary/cpu_kernel.hh"
#include "../kernels/simple_unary/cuda_kernel.hh"
#include "../kernels/simple_unary/cudnn_activation_kernel.hh"
#include "../kernels/simple_unary/cnnl_activation_kernel.hh"
#include "../kernels/simple_unary/cnnl_simple_unary_kernel.hh"
#include "common.h"

namespace refactor::kernel {

#define CASE(OP)              \
    case SimpleUnaryType::OP: \
        return #OP

    std::string_view unaryName(SimpleUnaryType type) {
        switch (type) {
            CASE(Abs);
            CASE(Acos);
            CASE(Acosh);
            CASE(Asin);
            CASE(Asinh);
            CASE(Atan);
            CASE(Atanh);
            CASE(Cos);
            CASE(Cosh);
            CASE(Sin);
            CASE(Sinh);
            CASE(Tan);
            CASE(Tanh);
            CASE(Relu);
            CASE(Sqrt);
            CASE(Sigmoid);
            CASE(Erf);
            CASE(Neg);
            CASE(Not);
            CASE(HardSwish);
            default:
                UNREACHABLE();
        }
    }

#define REGISTER(T)                          \
    if (auto ptr = T::build(type, a); ptr) { \
        ans.emplace_back(std::move(ptr));    \
    }

    std::vector<KernelBox>
    SimpleUnaryCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &a = inputs[0];

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                REGISTER(SimpleUnaryCpu)
                break;
            case decltype(_target)::Nvidia:
                REGISTER(ActivationCudnn)
                REGISTER(SimpleUnaryCuda)
                break;
            case decltype(_target)::Mlu:
                REGISTER(ActivationCnnl)
                REGISTER(SimpleUnaryCnnl)
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
