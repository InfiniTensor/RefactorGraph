#include "kernel/attributes/slice_info.h"
#include <numeric>

namespace refactor::kernel {

    bool SliceInfo::Dim::operator==(Dim const &rhs) const noexcept {
        return countStride == rhs.countStride &&
               sizeStart == rhs.sizeStart &&
               sizeStride == rhs.sizeStride;
    }
    bool SliceInfo::Dim::operator!=(Dim const &rhs) const noexcept {
        return !operator==(rhs);
    }

    SliceInfo::SliceInfo(
        std::vector<Dim> dims_,
        dim_t blockCount_,
        dim_t blockSize_,
        dim_t baseOffset_) noexcept
        : dims(std::move(dims_)),
          blockCount(blockCount_),
          blockSize(blockSize_),
          baseOffset(baseOffset_) {}

    SliceInfo::SliceInfo(Dimensions const &dims_, Tensor const &input)
        : dims(1),
          blockCount(1),
          blockSize(input.dataType.size()),
          baseOffset(0) {
        ASSERT(dims_.size() == static_cast<size_t>(input.rank()), "Unreachable");

        auto continuous = true;
        auto stride = blockSize;
        dims[0] = {1, 0, static_cast<sdim_t>(stride)};
        for (auto i : range0_(input.rank()).rev()) {
            auto l = input.shape[i];
            auto const &d = dims_[i];
            if (auto &it = dims.back(); continuous && d.step == 1) {
                it.countStride *= d.length;
                it.sizeStart = d.start * stride;
                it.sizeStride *= l;
            } else {
                dims.push_back(Dim{
                    static_cast<dim_t>(it.countStride * d.length),
                    static_cast<dim_t>(d.start * stride),
                    static_cast<sdim_t>(d.step * stride),
                });
            }
            continuous = d.length == l;
            stride *= l;
        }
        baseOffset = dims[0].sizeStart;
        auto elementCount = dims[0].countStride;
        blockSize *= elementCount;
        for (auto &d : dims) {
            d.countStride /= elementCount;
        }
        std::reverse(dims.begin(), dims.end());
        blockCount = dims[0].countStride;
        for (auto i : range(1ul, dims.size())) {
            dims[i - 1].countStride = dims[i].countStride;
        }
        dims.pop_back();
        dims.shrink_to_fit();
    }

    SliceInfo SliceInfo::reform(dim_t maxblockSize) const noexcept {
        auto blockSize_ = std::gcd(blockSize, maxblockSize);
        if (blockSize_ == blockSize) { return *this; }
        auto times = blockSize / blockSize_;
        SliceInfo ans{
            std::vector<Dim>(dims.size() + 1),
            blockCount * times,
            blockSize_,
            baseOffset,
        };
        for (auto i : range0_(dims.size())) {
            auto const &d = dims[i];
            ans.dims[i] = {
                d.countStride * times,
                d.sizeStart,
                d.sizeStride,
            };
        }
        ans.dims.back() = {1, 0, static_cast<sdim_t>(blockSize_)};
        return ans;
    }

    void SliceInfo::reformAssign(dim_t maxblockSize) noexcept {
        auto blockSize_ = std::gcd(blockSize, maxblockSize);
        if (blockSize_ == blockSize) { return; }
        auto times = blockSize / blockSize_;
        blockCount *= times;
        blockSize = blockSize_;
        for (auto &d : dims) {
            d.countStride *= times;
        }
        dims.resize(dims.size() + 1);
        dims.back() = {1, 0, static_cast<sdim_t>(blockSize_)};
    }


}// namespace refactor::kernel
