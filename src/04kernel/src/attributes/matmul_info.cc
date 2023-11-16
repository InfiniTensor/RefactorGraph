#include "kernel/attributes/matmul_info.h"
#include <numeric>
namespace refactor::kernel {

    ExpandInfo buildBias(size_t m, size_t n,
                         Tensor const &a,
                         Tensor const &b,
                         Tensor const &c) {
        std::vector<dim_t> output(std::max(a.rank(), b.rank()));
        for (auto i : range0_(output.size() - 2)) {
            auto a_ = i < a.rank() ? a.shape[i] : 1;
            auto b_ = i < b.rank() ? b.shape[i] : 1;
            output[i] = std::max(a_, b_);
        }
        output.rbegin()[1] = m;
        output.rbegin()[0] = n;

        return ExpandInfo(
            c.dataType,
            slice(c.shape.data(), c.rank()),
            slice(output.data(), output.size()));
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
          biasExpand(c ? std::make_optional(buildBias(m, n, a, b, *c)) : std::nullopt),
          broadcaster({
              slice(a.shape.data(), a.shape.size() - 2),
              slice(b.shape.data(), b.shape.size() - 2),
          }) {
        auto kB = transB ? b.shape.rbegin()[0] : b.shape.rbegin()[1];
        ASSERT(k == kB, "MatMul: input shape not matched.");
    }

}// namespace refactor::kernel
