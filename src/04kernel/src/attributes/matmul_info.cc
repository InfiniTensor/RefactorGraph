#include "kernel/attributes/matmul_info.h"
#include <numeric>
namespace refactor::kernel {
    Broadcaster inferBroadcast(Shape const &A, Shape const &B) {
        ASSERT(A.size() >= 2 && B.size() >= 2, "MatMul: invalid shape broadcast.");
        size_t rankA = A.size() - 2;
        size_t rankB = B.size() - 2;
        auto A_ = Shape(A.begin(), A.begin() + rankA);
        auto B_ = Shape(B.begin(), B.begin() + rankB);
        return Broadcaster({A_, B_});
    }

    void inferBMKN(MatMulInfo *info, Tensor const &A, Tensor const &B) {
        info->b = info->broadcaster.outputsCount;
        auto kA = *(info->transA ? A.shape.rbegin() + 1 : A.shape.rbegin());
        auto kB = *(info->transB ? B.shape.rbegin() : B.shape.rbegin() + 1);
        ASSERT(kA == kB, "MatMul: input shape not matched.");
        info->m = *(info->transA ? A.shape.rbegin() : A.shape.rbegin() + 1);
        info->n = *(info->transB ? B.shape.rbegin() + 1 : B.shape.rbegin());
        info->k = kA;
    }

    void setBiasMode(MatMulInfo *info, Tensor const &C) {
        if (C.elementsSize() == info->m * info->n) {
            info->biasMode = BiasBroadcast::NONE;
        } else if (C.shape.size() == 0 || C.elementsSize() == 1) {
            info->biasMode = BiasBroadcast::CONST;
        } else if (C.shape.back() == info->n) {
            info->biasMode = BiasBroadcast::ROW;
        } else {
            info->biasMode = BiasBroadcast::COL;
        }
    }

    MatMulInfo::MatMulInfo(Tensor const &A, Tensor const &B, bool tA, bool tB,
                           float a, float b)
        : dataType(A.dataType), transA(tA), transB(tB), useBias(false), alpha(a), beta(b), broadcaster(inferBroadcast(A.shape, B.shape)) {
        inferBMKN(this, A, B);
    }

    MatMulInfo::MatMulInfo(Tensor const &A, Tensor const &B, Tensor const &C, bool tA, bool tB,
                           float a, float b)
        : dataType(C.dataType), transA(tA), transB(tB), useBias(true), alpha(a), beta(b), broadcaster(inferBroadcast(A.shape, B.shape)) {
        inferBMKN(this, A, B);
        setBiasMode(this, C);
    }


}// namespace refactor::kernel
