#include "kernel/attributes/pad_info.h"
#include <iostream>
#include <numeric>

namespace refactor::kernel {

    PadInfo::PadInfo(
        PadsShape pads_,
        PadType mode_,
        Tensor const &x,
        Tensor const &y,
        bool have_value_) noexcept : rank(x.rank()), mode(mode_), pads(std::move(pads_)), wholeNDim(rank, 0),
                                     partNDim(rank, 0), partStride(rank, 1), type(x.dataType), have_value(have_value_),
                                     size(0) {
        int64_t p = 1;
        for (auto i = rank - 1; i >= 0; --i) {
            wholeNDim[i] = y.shape[i];
            partNDim[i] = x.shape[i];
            partStride[i] = p;
            p = p * partNDim[i];
        }
        size = std::accumulate(wholeNDim.begin(), wholeNDim.end(), 1, std::multiplies<>());
    }

}// namespace refactor::kernel
