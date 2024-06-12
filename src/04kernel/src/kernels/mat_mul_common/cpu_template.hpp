#ifndef KERNEL_MATMUL_COMMON_CPU_TEMPLATE_HPP
#define KERNEL_MATMUL_COMMON_CPU_TEMPLATE_HPP

namespace refactor::kernel {

    template<class TO, class TI>
    struct MatMulCPUMetaData {
        size_t M, K, N,
            strideA0, strideA1,
            strideB0, strideB1;
        TI alpha;
        TO beta;

        /*
         * 2D matrix multiplication: Y = a * A @ B + b * Y
         * Assume bias C has been broadcast to Y already. Beta should be 0 in the absence of bias.
         */
        void matrixMultiply(TI const *a, TI const *b, TO *y) const noexcept {
#pragma omp parallel for simd
            for (size_t ind = 0; ind < M * N; ind++) {
                size_t i = ind / N;
                size_t j = ind % N;
                TO sum = 0;

                for (size_t k = 0; k < K; k++) {
                    sum += static_cast<TO>(a[i * strideA0 + k * strideA1] * b[k * strideB0 + j * strideB1]);
                }
                y[ind] = beta * y[ind] + alpha * sum;
            }
        }
    };

}// namespace refactor::kernel

#endif// KERNEL_MATMUL_COMMON_CPU_TEMPLATE_HPP
