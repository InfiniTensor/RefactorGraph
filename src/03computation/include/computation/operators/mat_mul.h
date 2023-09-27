#ifndef COMPUTATION_MAT_MUL_H
#define COMPUTATION_MAT_MUL_H

#include "../operator.h"

namespace refactor::computation {

    struct MatMul : public Operator {
        float alpha, beta;
        bool transA, transB;

        constexpr MatMul(float alpha_, float beta_, bool transA_, bool transB_)
            : Operator(), alpha(alpha_), beta(beta_), transA(transA_), transB(transB_) {}
    };

}// namespace refactor::computation

#endif// COMPUTATION_MAT_MUL_H
