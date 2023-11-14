#include "kernel/attributes/gather_info.h"
#include <numeric>

namespace refactor::kernel {
    /// Gather 算子的语义逻辑见 <https://onnx.ai/onnx/operators/onnx__Gather.html>。
    ///
    /// `data` 的形状为 `[x0, x1,..., xa, xa+1,..., xn-1]`，`indices` 的形状为 `[i0, i1, ..,im-1]`。
    /// 设 `data` 的 `axis` 轴为 `xa`，则 `indices` 中存储的数据用于选择 `data` 中 `xa` 维度的值。
    /// Gather 计算时，`xa` 之后的维度都是连续的，而 `xa` 之前的维度顺序保持不变。
    ///
    /// 因此，只需要保存这些参数：
    ///
    /// | 字段名      | 计算式                                    | 说明
    /// | ---------- | ---------------------------------------- | ------
    /// | `prefix`   | π i∈[0,a) xi = x0 * x1 * ... * xa-1     | `xa` 之前维度打包
    /// | `postfix`  | π i∈(a,n) xi = xa+1 * xa+2 * ... * xn-1 | `xa` 之后维度打包
    /// | `midSizeI` | xa                                       | `xa` 维度重排前的大小
    /// | `midSizeO` | π t∈[0,m) it = i0 * i1 * ... * im-1     | `xa` 维度重排后的大小，即 `indices` 形状打包
    GatherInfo::GatherInfo(dim_t axis, Tensor const &data, Tensor const &indices) noexcept
        : prefix(0), postfix(0), midSizeI(0), midSizeO(0), idxType(indices.dataType) {

        auto axisIt = data.shape.begin() + axis;
        prefix = std::accumulate(data.shape.begin(), axisIt, 1, std::multiplies());
        midSizeI = *axisIt++;
        postfix = std::accumulate(axisIt, data.shape.end(), data.dataType.size(), std::multiplies());
        midSizeO = std::accumulate(indices.shape.begin(), indices.shape.end(), 1, std::multiplies());
    }

}// namespace refactor::kernel
