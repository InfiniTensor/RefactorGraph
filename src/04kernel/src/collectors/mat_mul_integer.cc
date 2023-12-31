﻿#include "kernel/collectors/mat_mul_integer.h"
#include "../../src/kernels/mat_mul_integer/cpu_kernel.hh"
#include "../../src/kernels/mat_mul_integer/cublas_kernel.hh"
#include "kernel/attributes/mat_mul_integer_info.h"

namespace refactor::kernel {

    std::vector<KernelBox>
    MatMulIntegerCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        MatMulIntegerInfo info(inputs);

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                if (auto ptr = MatMulIntegerCpu::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            case decltype(_target)::Nvidia:
                if (auto ptr = MatMulIntegerCublas::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
