#include "kernel/attributes/transpose_info.h"

namespace refactor::kernel {

    TransposeInfo::TransposeInfo(Shape const &shape, Permutation const &perm) noexcept
        : dims(), size(0) {
        auto rank = shape.size();
        ASSERT(perm.size() == rank, "");
        if (rank == 0) { return; }

        // 对 permutation 分段，每个段里是一些连续的维度号
        // 第 i 个 perm 到其所在的第 j 个段的映射
        constexpr static auto INVALID = std::numeric_limits<dim_t>::max();
        std::vector<dim_t> segOfDims(rank, 0);
        dim_t segs = 0, lastPerm = perm[0];
        for (auto i : range(1ul, rank)) {
            if (auto p = perm[i]; p == lastPerm + 1) {
                // 维度连续，不分段
                segOfDims[lastPerm = p] = segs;
            } else if (shape[p] != 1) {
                // 维度不连续，形状不为 1，分段
                segOfDims[lastPerm = p] = ++segs;
            } else {
                // 形状 1 不会导致分段
                segOfDims[p] = INVALID;
            }
        }

        // 整理出根据输入维度顺序排列的、合并后的各维度形状和输出位置。
        struct SizePerm {
            dim_t sizeI, perm, sizeO, strideI, strideO;
        };
        absl::InlinedVector<SizePerm, 4> forward(++segs);
        dim_t j = 0;
        auto seg = INVALID;
        for (auto i : range0_(rank)) {
            if (segOfDims[i] == INVALID) { continue; }
            auto last = std::exchange(seg, segOfDims[i]);
            if (seg != last) {
                forward[j++] = {shape[i], seg, 1, 1, 1};
            } else {
                forward[j - 1].sizeI *= shape[i];
            }
        }

        for (auto i : range0_(segs)) {
            forward[forward[i].perm].sizeO = forward[i].sizeI;
        }
        for (auto i : range(1u, segs).rev()) {
            forward[i - 1].strideI = forward[i].strideI * forward[i].sizeI;
            forward[i - 1].strideO = forward[i].strideO * forward[i].sizeO;
        }

        dims.assign(segs, {});
        for (auto i : range0_(segs)) {
            dims[i].strideO = forward[i].strideO;
            dims[forward[i].perm].strideI = forward[i].strideI;
        }
        size = forward[0].sizeI * forward[0].strideI;
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

}// namespace refactor::kernel
