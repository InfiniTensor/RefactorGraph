#ifndef COMPUTATION_PASS_MANAGER_H
#define COMPUTATION_PASS_MANAGER_H
#include "conv_to_matmul.h"
#include "convert.h"
#include "matmul_transpose.h"

namespace refactor::computation {

    void register_() {
#define REGISTER(PASS, NAME) static ConverterRegister<PASS> NAME("" #NAME);
        REGISTER(MatMulTransposeFuse, MatMulTransposeFuse)
        REGISTER(ConvToMatmul, ConvToMatmul)
    };


}// namespace refactor::computation

#endif