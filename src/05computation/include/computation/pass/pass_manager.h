#ifndef COMPUTATION_PASS_MANAGER_H
#define COMPUTATION_PASS_MANAGER_H
#include "convert.h"
#include "matmul_transpose.h"

namespace refactor::computation {

    void register_() {
#define REGISTER(PASS, NAME) static ConverterRegister<PASS> __l("" #NAME);
        REGISTER(MatMulTransposeFuse, MatMulTransposeFuse)
    };


}// namespace refactor::computation

#endif