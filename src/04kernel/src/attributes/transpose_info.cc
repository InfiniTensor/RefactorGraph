#include "kernel/attributes/transpose_info.h"

namespace refactor::kernel {

    TransposeInfo::TransposeInfo(Shape const &shape, Permutation const &perm)
        : dims(), size(0) {
        auto rank = shape.size();
        ASSERT(perm.size() == rank, "");
        if (rank == 0) { return; }

        // 对 permutation 分段，每个段里是一些连续的维度号
        // 第 i 个 perm 到其所在的第 j 个段的映射
        std::vector<uint_lv2> dims_(rank, 0);
        uint_lv2 segs = 0;
        for (auto i : range(1ul, rank)) {
            dims_[perm[i]] = (shape[i] != 1 && perm[i] != perm[i - 1] + 1) ? ++segs : segs;
        }

        dims.resize(++segs, {1, dims_[0]});
        absl::InlinedVector<uint_lv2, 4> size_(segs, shape[0]);
        uint_lv2 j = 0;
        for (auto i : range(1ul, rank)) {
            if (dims_[i] != dims_[i - 1]) {
                dims[++j].permutation = dims_[i];
                size_[j] = shape[i];
            } else {
                size_[j] *= shape[i];
            }
        }

        for (auto i : range(1ul, dims.size()).rev()) {
            dims[i - 1].stride = dims[i].stride * size_[i];
        }
        size = dims[0].stride * size_[0];
    }

}// namespace refactor::kernel
