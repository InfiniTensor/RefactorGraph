#include "kernel/collectors/simple_binary.h"
#include "../kernels/simple_binary/arthimetic11.hh"
#include "../kernels/simple_binary/arthimetic11_cuda.hh"
#include "common/error_handler.h"

namespace refactor::kernel {

    std::vector<KernelBox>
    SimpleBinaryCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        std::vector<KernelBox> result;
        auto const &a = inputs[0].get();
        auto const &b = inputs[1].get();
        auto const &c = outputs[0].get();

#define REGISTER(T)                                \
    if (auto box = T::build(type, a, b, c); box) { \
        result.emplace_back(std::move(box));       \
    }

        REGISTER(Arthimetic11)
        REGISTER(Arthimetic11Cuda)

        return result;
    }

}// namespace refactor::kernel
