#include "pad_2d_info.h"
#include <numeric>

namespace refactor::kernel {

    Pad2DInfo::Pad2DInfo(DataType dt, slice_t<dim_t> input, ddim_t const *pads)
        : blockCount(std::accumulate(input.begin(), input.end() - 2, 1, std::multiplies<>())),
          blockSize(dt.size()),
          hw(input.end()[-1] * input.end()[-2]),
          w(input.end()[-1]),
          padHW(pads[0] - pads[2]),
          padW(pads[1] - pads[3]) {}

}// namespace refactor::kernel
