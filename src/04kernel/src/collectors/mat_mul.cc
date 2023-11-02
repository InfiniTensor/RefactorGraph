#include "kernel/collectors/mat_mul.h"
#include "../kernels/mat_mul/cpu_kernel.hh"
#include "../kernels/mat_mul/cublas_kernel.hh"
#include "common.h"
#include "kernel/attributes/matmul_info.h"

namespace refactor::kernel {
#define REGISTER(T)                                \
    if (auto ptr = T::build(a, b, y, info); ptr) { \
        ans.emplace_back(std::move(ptr));          \
    }

    std::vector<KernelBox>
    MatMulCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &a = inputs[0].get();
        auto const &b = inputs[1].get();
        auto const &y = outputs[0].get();

        ASSERT(inputs.size() == 2 || inputs.size() == 3, "MatMul: invalid number of inputs.");

        auto info = inputs.size() == 3 ? MatMulInfo(a, b, inputs[2].get(), transA, transB, alpha, beta)
                                       : MatMulInfo(a, b, transA, transB, alpha);

        std::vector<KernelBox> ans;
        switch (target) {
            case Target::Cpu:
                REGISTER(MatMulCPU)
                break;
            case Target::NvidiaGpu:
                REGISTER(MatMulCublas)
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
