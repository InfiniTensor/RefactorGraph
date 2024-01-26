#include "kernel/collectors/simple_binary.h"
#include "../kernels/simple_binary/binary_cudnn.hh"
#include "../kernels/simple_binary/cpu_kernel.hh"
#include "../kernels/simple_binary/cuda_kernel.hh"

namespace refactor::kernel {

#define CASE(OP)               \
    case SimpleBinaryType::OP: \
        return #OP

    std::string_view opName(SimpleBinaryType type) {
        switch (type) {
            CASE(Add);
            CASE(Sub);
            CASE(Mul);
            CASE(Div);
            CASE(Pow);
            CASE(And);
            CASE(Or);
            CASE(Xor);
            CASE(Mod);
            CASE(Fmod);
            default:
                UNREACHABLE();
        }
    }

#define REGISTER(T)                             \
    if (auto ptr = T::build(type, a, b); ptr) { \
        ans.emplace_back(std::move(ptr));       \
    }
#define REGISTER_BROCAST(T)                        \
    if (auto ptr = T::build(type, a, b, c); ptr) { \
        ans.emplace_back(std::move(ptr));          \
    }

    std::vector<KernelBox>
    SimpleBinaryCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &a = inputs[0];
        auto const &b = inputs[1];
        auto const &c = outputs[0];

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                REGISTER(BinaryCpu)
                break;
            case decltype(_target)::Nvidia:
                REGISTER_BROCAST(BinaryCudnn)
                REGISTER(BinaryCuda)
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
