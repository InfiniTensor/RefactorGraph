#include "kernel/attributes/gather_info.h"
#include <numeric>

namespace refactor::kernel {

    GatherInfo::GatherInfo(uint_lv2 axis, Tensor const &data, Tensor const &indices) noexcept
        : prefix(0),
          postfix(0),
          midSizeI(data.shape[axis]),
          midSizeO(0),
          idxType(indices.dataType) {
        auto eleSize = data.dataType.size();
        auto axisIt = data.shape.begin() + axis;
        prefix = std::accumulate(data.shape.begin(), axisIt++, 1, std::multiplies<>());
        postfix = std::accumulate(axisIt, data.shape.end(), eleSize, std::multiplies<>());
        midSizeO = std::accumulate(indices.shape.begin(), indices.shape.end(), 1, std::multiplies<>());
    }

    int64_t GatherInfo::index(void const *indices, uint_lv2 i) const noexcept {
        auto ptr = reinterpret_cast<uint8_t const *>(indices);
        switch (idxType) {
            case DataType::I64:
                return reinterpret_cast<int64_t const *>(ptr)[i];
            case DataType::I32:
                return reinterpret_cast<int32_t const *>(ptr)[i];
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
