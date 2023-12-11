#include "kernel/attributes/scatter_nd_info.h"
#include <numeric>

namespace refactor::kernel {

#define K (indices.shape.back())

    ScatterNDInfo::ScatterNDInfo(
        Tensor const &data,
        Tensor const &indices)
        : prefix(std::accumulate(
              indices.shape.begin(),
              indices.shape.begin() + indices.shape.size() - 1,
              1,
              std::multiplies())),
          strides(K, 1),
          blockSize(std::accumulate(
              data.shape.begin() + K,
              data.shape.end(),
              data.dataType.size(),
              std::multiplies())) {
        for (auto i : range0_(K - 1).rev()) {
            strides[i] = strides[i + 1] * data.shape[i + 1];
        }
    }

}// namespace refactor::kernel
