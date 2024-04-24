#include "kernel/attributes/transpose_info.h"
#include <numeric>
#include <span>

namespace refactor::kernel {

    TransposeInfo::TransposeInfo(DataType dt, Shape const &shape_, Permutation const &perm_)
        : dims{},
          blockSize(dt.size()),
          blockCount(std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies())) {
        auto rank = shape_.size();
        ASSERT(perm_.size() == rank, "");

        std::vector<dim_t> shape;
        std::vector<dim_t> perm;

        // 去除形状里的 1 维度
        {
            shape.reserve(rank);
            std::vector<ddim_t> mapDim(rank, 0);
            for (dim_t sub = 0; auto i : range0_(rank)) {
                if (auto l = shape_[i]; l != 1) {
                    shape.push_back(l);
                    mapDim[i] = i - sub;
                } else {
                    ++sub;
                    mapDim[i] = -1;
                }
            }

            perm.reserve(rank = shape.size());
            for (auto dim : perm_) {
                if (auto to = mapDim[dim]; to >= 0) {
                    perm.push_back(to);
                }
            }
        }
        if (rank <= 1) {
            dims = {{1, 1}};
            blockSize *= blockCount;
            blockCount = 1;
            return;
        }
        // 合并连续的维度
        {
            std::vector<ddim_t> mapDim(rank, 0);
            std::iota(mapDim.begin(), mapDim.end(), 0);

            for (auto past = perm[0]; auto dim : std::span(perm.begin() + 1, perm.end())) {
                if (dim == past + 1) {
                    mapDim[dim] = -1;
                    for (auto j : range<dim_t>(dim + 1, rank)) {
                        --mapDim[j];
                    }
                }
                past = dim;
            }

            auto j = 0;
            for (auto i : range(1ul, rank)) {
                if (mapDim[i] >= 0) {
                    shape[++j] = shape[i];
                } else {
                    shape[j] *= shape[i];
                }
            }
            shape.resize(rank = j + 1);

            for (auto i = 0; auto from : perm) {
                if (auto to = mapDim[from]; to >= 0) {
                    perm[i++] = to;
                }
            }
            perm.resize(rank);
        }
        if (rank <= 1) {
            dims = {{1, 1}};
            blockSize *= blockCount;
            blockCount = 1;
            return;
        }
        // 合并末尾连续访存
        if (perm.back() == rank - 1) {
            blockSize *= shape.back();
            blockCount /= shape.back();
            shape.pop_back();
            perm.pop_back();
            --rank;
        }
        // 计算 stride
        struct StrideI {
            dim_t strideI;
        };
        std::vector<StrideI> buf(rank, {1});
        std::vector<dim_t> permMap(rank);
        dims.resize(rank, {1, 1});
        for (auto i : range(1ul, rank).rev()) {
            permMap[perm[i]] = i;
            // clang-format off
             buf[i - 1].strideI =  buf[i].strideI * shape[     i ];
            dims[i - 1].strideO = dims[i].strideO * shape[perm[i]];
            // clang-format on
        }
        // 输入的 stride 按输出顺序重排
        for (auto i : range0_(rank)) {
            dims[i].strideI = buf[permMap[i]].strideI;
        }
    }

    dim_t TransposeInfo::locate(dim_t i) const noexcept {
        dim_t ans = 0;
        long rem = i;
        for (auto [strideI, strideO] : dims) {
            auto d = std::div(rem, strideO);
            ans += d.quot * strideI;
            rem = d.rem;
        }
        return ans;
    }

    TransposeInfo TransposeInfo::reform(dim_t maxblockSize) const noexcept {
        auto ans = *this;
        ans.reformAssign(maxblockSize);
        return ans;
    }
    void TransposeInfo::reformAssign(dim_t maxblockSize) noexcept {
        if (dims.empty()) { return; }
        auto blockSize_ = std::gcd(blockSize, maxblockSize);
        if (blockSize_ == blockSize) { return; }
        auto times = blockSize / blockSize_;
        blockCount *= times;
        blockSize = blockSize_;
        for (auto &s : dims) {
            s.strideI *= times;
            s.strideO *= times;
        }
        dims.resize(dims.size() + 1);
        dims.rbegin()[0] = {1, 1};
    }

}// namespace refactor::kernel
