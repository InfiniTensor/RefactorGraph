#include "kernel/attributes/slice_info.h"
#include <numeric>

namespace refactor::kernel {

    bool SliceInfo::Dim::operator==(Dim const &rhs) const noexcept {
        return strideO == rhs.strideO &&
               strideI == rhs.strideI &&
               skip == rhs.skip;
    }
    bool SliceInfo::Dim::operator!=(Dim const &rhs) const noexcept {
        return !operator==(rhs);
    }

    SliceInfo::SliceInfo(
        std::vector<Dim> dims_,
        dim_t blockCount_,
        dim_t blockSize_) noexcept
        : dims(std::move(dims_)),
          blockCount(blockCount_),
          blockSize(blockSize_) {}

    SliceInfo::SliceInfo(Dimensions dims_, Tensor const &input)
        : dims{},
          blockCount(1),
          blockSize(input.dataType.size()) {
        size_t rank = input.rank();
        if (!rank) { return; }// scalar input
        ASSERT(dims_.size() == rank, "unreachable");

        std::vector<dim_t> shape;
        {// 去除形状里的 1
            shape.reserve(rank);
            for (auto i : range0_(rank)) {
                if (auto l = input.shape[i]; l != 1) {
                    if (auto j = shape.size(); j < i) { dims_[j] = dims_[i]; }
                    shape.push_back(l);
                }
            }
            dims_.resize(rank = shape.size());
        }
        dims.reserve(rank);
        dim_t strideI = 1;
        for (auto i : range0_(rank).rev()) {
            auto const &dim = dims_[i];
            dims.push_back({
                .strideO = blockCount,
                .skip = static_cast<dim_t>(strideI * dim.start),
                .strideI = static_cast<sdim_t>(strideI * dim.step),
            });
            blockCount *= dim.length;
            strideI *= shape[i];
        }
        std::reverse(dims.begin(), dims.end());

        while (!dims.empty()) {
            auto const &dim = dims.back();
            if (dim.strideI == static_cast<sdim_t>(dim.strideO) && !dim.skip) {
                dims.pop_back();
            } else {
                long times = std::gcd(std::gcd(dim.strideI, dim.strideO), dim.skip);
                blockCount /= times;
                blockSize *= times;
                if (!dims.empty()) {
                    for (auto &dim : dims) {
                        dim.strideO /= times;
                        dim.skip /= times;
                        dim.strideI /= times;
                    }
                    if (dims.back().strideO != 1) {
                        dims.push_back({1, 0, 1});
                    }
                }
                break;
            }
        }
    }

    SliceInfo SliceInfo::reform(dim_t maxblockSize) const noexcept {
        auto ans = *this;
        ans.reformAssign(maxblockSize);
        return ans;
    }

    void SliceInfo::reformAssign(dim_t maxblockSize) noexcept {
        auto blockSize_ = std::gcd(blockSize, maxblockSize);
        if (blockSize_ == blockSize) { return; }
        auto times = blockSize / blockSize_;
        blockCount *= times;
        blockSize = blockSize_;
        for (auto &d : dims) {
            d.strideO *= times;
            d.strideI *= times;
            d.skip *= times;
        }
        dims.resize(dims.size() + 1);
        dims.back() = {1, 0, 1};
    }


}// namespace refactor::kernel
