#ifndef COMPUTATION_PASS_REGISTER_H
#define COMPUTATION_PASS_REGISTER_H
#include "pass/conv_to_matmul.h"
#include "pass/converter.h"
#include "pass/layernorm_fuse.h"
#include "pass/matmul_transpose.h"

namespace refactor::computation {

    void register_() {
#define REGISTER(PASS, NAME) static ConverterRegister<PASS> NAME("" #NAME);
        REGISTER(MatMulTransposeFuse, MatMulTransposeFuse)
        REGISTER(ConvToMatmul, ConvToMatmul)
        REGISTER(LayernormFuse, LayernormFuse)
    };


}// namespace refactor::computation

#endif