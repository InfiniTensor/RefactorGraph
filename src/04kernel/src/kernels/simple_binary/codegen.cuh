#ifndef KERNEL_BINARY_CUDA_CODEGEN_CUH
#define KERNEL_BINARY_CUDA_CODEGEN_CUH

#include "kernel/collectors/simple_binary.h"

namespace refactor::kernel {

    constexpr std::string_view op(SimpleBinaryType op, DataType dt) {
        switch (op) {
            case SimpleBinaryType::Add:
                return "a + b";
            case SimpleBinaryType::Sub:
                return "a - b";
            case SimpleBinaryType::Mul:
                return "a * b";
            case SimpleBinaryType::Div:
                return "a / b";
            case SimpleBinaryType::And:
                return "a && b";
            case SimpleBinaryType::Or:
                return "a || b";
            case SimpleBinaryType::Xor:
                return "a ^ b";
            case SimpleBinaryType::Pow:
                switch (dt) {
                    case DataType::F32:
                        return "powf(a, b)";
                    case DataType::FP16:
                        return "__float2half(__powf(__half2float(a), __half2float(b)))";
                    case DataType::BF16:
                        return "__float2bfloat16(powf(__bfloat162float(a), __bfloat162float(b)))";
                    default:
                        return "pow(a, b)";
                }
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel

#endif// KERNEL_BINARY_CUDA_CODEGEN_CUH
