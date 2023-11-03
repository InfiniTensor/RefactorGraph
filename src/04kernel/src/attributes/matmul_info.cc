#include "kernel/attributes/matmul_info.h"
#include <numeric>
namespace refactor::kernel {

    BiasType buildBiasType(size_t m, size_t n, Tensor const &c) {
        switch (c.rank()) {
            case 0:
                return BiasType::Scalar;
            case 1:
                return c.shape[0] == n
                           ? BiasType::RowVector
                           : BiasType::Scalar;
            case 2:
                if (c.shape[0] == m) {
                    return c.shape[1] == n
                               ? BiasType::Matrix
                               : BiasType::ColVector;
                } else {
                    return c.shape[1] == n
                               ? BiasType::RowVector
                               : BiasType::Scalar;
                }
            default:
                UNREACHABLE();
        }
    }

    MatMulInfo::MatMulInfo(
        Tensor const &a, Tensor const &b,
        std::optional<std::reference_wrapper<Tensor const>> c,
        bool transA_, bool transB_, float alpha_, float beta_)
        : dataType(a.dataType),
          alpha(alpha_), beta(beta_),
          transA(transA_), transB(transB_),
          m(transA ? a.shape.rbegin()[0] : a.shape.rbegin()[1]),
          n(transB ? b.shape.rbegin()[1] : b.shape.rbegin()[0]),
          k(transA ? a.shape.rbegin()[1] : a.shape.rbegin()[0]),
          biasType(c ? buildBiasType(m, n, *c) : BiasType::NoBias),
          broadcaster({
              slice(a.shape.data(), a.shape.size() - 2),
              slice(b.shape.data(), b.shape.size() - 2),
          }) {
        auto kB = transB ? b.shape.rbegin()[0] : b.shape.rbegin()[1];
        ASSERT(k == kB, "MatMul: input shape not matched.");
    }

    MatMulInfo::MatMulInfo(Tensor const &A, Tensor const &B, bool tA, bool tB, float alpha)
        : MatMulInfo(A, B, {}, tA, tB, alpha, 0.f) {}

    MatMulInfo::MatMulInfo(Tensor const &A, Tensor const &B, Tensor const &C, bool tA, bool tB, float alpha, float beta)
        : MatMulInfo(A, B, std::make_optional(C), tA, tB, alpha, beta) {}

    size_t MatMulInfo::batch() const noexcept {
        return broadcaster.outputsCount;
    }

}// namespace refactor::kernel
