#include "kernel/attributes/softmax_info.h"
#include <numeric>

namespace refactor::kernel {

    SoftmaxInfo::SoftmaxInfo(Tensor const &data, uint_lv2 axis) noexcept
        : pre(0), mid(0), post(0), type(data.dataType) {

        auto axisIt = data.shape.begin() + axis;
        pre = std::accumulate(data.shape.begin(), axisIt, 1, std::multiplies<>());
        mid = *axisIt++;
        post = std::accumulate(axisIt, data.shape.end(), 1, std::multiplies<>());
    };

}// namespace refactor::kernel
