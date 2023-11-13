#include "kernel/attributes/slice_info.h"

namespace refactor::kernel {

    bool SliceInfo::Dim::operator==(Dim const &rhs) const noexcept {
        return countStride == rhs.countStride &&
               sizeStart == rhs.sizeStart &&
               sizeStride == rhs.sizeStride;
    }
    bool SliceInfo::Dim::operator!=(Dim const &rhs) const noexcept {
        return !operator==(rhs);
    }

    SliceInfo::SliceInfo(Dimensions const &dims_, Tensor const &input) noexcept
        : blockSize(input.dataType.size()), dims(1) {
        ASSERT(dims_.size() == input.rank(), "Unreachable");

        auto continuous = true;
        auto stride = blockSize;
        dims[0] = {1, 0, static_cast<sdim_t>(stride)};
        for (auto i : range0_(input.rank()).rev()) {
            auto l = input.shape[i];
            auto const &d = dims_[i];
            if (continuous && d.step == 1) {
                auto &it = dims.back();
                it.countStride *= d.length;
                it.sizeStart = d.start * stride;
                it.sizeStride *= l;
            } else {
                dims.push_back(Dim{
                    static_cast<dim_t>(dims.back().countStride * d.length),
                    static_cast<dim_t>(d.start * stride),
                    static_cast<sdim_t>(d.step * stride),
                });
            }
            continuous = d.length == l;
            stride *= l;
        }
        auto blockCount = dims[0].countStride;
        blockSize *= blockCount;
        for (auto &d : dims) {
            d.countStride /= blockCount;
        }
        std::reverse(dims.begin(), dims.end());
    }

}// namespace refactor::kernel
