#include "kernel/collectors/simple_binary.h"
#include "../kernels/simple_binary/basic_cpu.hh"
#include "../kernels/simple_binary/basic_cuda.hh"
#include "../kernels/simple_binary/binary_cudnn.hh"
#include "../kernels/simple_binary/no_broadcast_cpu.hh"
#include "../kernels/simple_binary/no_broadcast_cuda.hh"

namespace refactor::kernel {

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
                REGISTER(Binary11Cpu)
                REGISTER(BinaryBasicCpu)
                break;
            case decltype(_target)::Nvidia:
                REGISTER_BROCAST(BinaryCudnn)
                REGISTER(Binary11Cuda)
                REGISTER(BinaryBasicCuda)
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
