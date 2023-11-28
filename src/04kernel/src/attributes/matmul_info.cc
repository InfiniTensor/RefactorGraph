#include "kernel/attributes/matmul_info.h"
#include <cstddef>
#include <numeric>

namespace refactor::kernel {

    ExpandInfo buildBias(size_t m, size_t n,
                         Tensor const &a,
                         Tensor const &b,
                         Tensor const &c) {
        std::vector<dim_t> output(std::max(a.rank(), b.rank()));
        auto it = output.rbegin();
        *it++ = n;
        *it++ = m;
        for (auto da = static_cast<size_t>(a.rank() - 2),
                  db = static_cast<size_t>(b.rank() - 2);
             auto i : range0_(output.size() - 2)) {
            auto a_ = i < da ? a.shape[da - i - 1] : 1;
            auto b_ = i < db ? b.shape[db - i - 1] : 1;
            *it++ = std::max(a_, b_);
        }

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
          k(transA ? a.shape.rbegin()[1] : a.shape.rbegin()[0]),
          n(transB ? b.shape.rbegin()[1] : b.shape.rbegin()[0]),
          biasExpand(c ? std::make_optional(buildBias(m, n, a, b, *c)) : std::nullopt),
          broadcaster({
              slice(a.shape.data(), a.shape.size() - 2),
              slice(b.shape.data(), b.shape.size() - 2),
          }) {
        auto kB = transB ? b.shape.rbegin()[0] : b.shape.rbegin()[1];
        ASSERT(k == kB, "MatMul: input shape not matched.");
    }

}// namespace refactor::kernel
