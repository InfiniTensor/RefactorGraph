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
        : blockCount(1),
          blockSize(input.dataType.size()),
          baseOffset(0),
          dims(1) {
        ASSERT(dims_.size() == input.rank(), "Unreachable");

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

}// namespace refactor::kernel
