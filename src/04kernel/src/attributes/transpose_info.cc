#include "kernel/attributes/transpose_info.h"

namespace refactor::kernel {

    TransposeInfo::TransposeInfo(Shape const &shape, Permutation const &perm)
        : _dims() {
        auto rank = shape.size();
        ASSERT(perm.size() == rank, "");
        if (rank == 0) { return; }

        std::vector<uint_lv2>// 对 permutation 分段，每个段里是一些连续的维度号
            segs{perm[0]},   // 第 j 个段到其在 perm 中的起始维度 i 的映射
            dims(rank, 0);   // 第 i 个 perm 到其所在的第 j 个段的映射
        auto r = range(1ul, rank);
        std::transform(r.begin(), r.end(), dims.begin() + 1,
                       [&](auto i) {
                           if (perm[i] != perm[i - 1] + 1) {
                               segs.push_back(perm[i]);
                           }
                           return segs.size() - 1;
                       });

        constexpr static auto INVALID = std::numeric_limits<uint_lv2>::max();
        _dims.resize(segs.size(), {1, INVALID});
        for (auto i : range0_(rank)) {
            auto j = dims[i];
            if (_dims[j].permutation == INVALID) {
                _dims[j].permutation = std::distance(segs.begin(), std::find(segs.begin(), segs.end(), j));
            }
            _dims[j].size *= shape[perm[i]];
        }
    }

}// namespace refactor::kernel
