#include "kernel/collectors/mat_mul.h"
#include "../kernels/mat_mul/cnnl_kernel.hh"
#include "../kernels/mat_mul/cpu_kernel.hh"
#include "../kernels/mat_mul/cublas_kernel.hh"
#include "kernel/attributes/mat_mul_info.h"

namespace refactor::kernel {
#define REGISTER(T)                       \
    if (auto ptr = T::build(info); ptr) { \
        ans.emplace_back(std::move(ptr)); \
    }

    std::vector<KernelBox>
    MatMulCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &a = inputs[0];
        auto const &b = inputs[1];

        auto info = inputs.size() == 3
                        ? MatMulInfo(a, b, std::make_optional(inputs[2]), transA, transB, alpha, beta)
                        : MatMulInfo(a, b, std::nullopt, transA, transB, alpha, beta);

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                REGISTER(MatMulCPU)
                break;
            case decltype(_target)::Nvidia:
                REGISTER(MatMulCublas)
                break;
            case decltype(_target)::Mlu:
                if (auto ptr = MatMulCnnl::build(inputs, outputs, transA, transB, alpha, beta); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
