#include "kernel/attributes/broadcaster.h"
#include <numeric>

namespace refactor::kernel {
    /// 多向广播的语义逻辑见 <https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md#multidirectional-broadcasting>。
    ///
    /// 广播器在一次循环中构造各维度的步长，并执行 2 种优化：
    ///
    /// - 消除所有输入全为 1 的维度；
    /// - 尽量合并相邻的维度；
    ///   > 相邻维度可以合并的条件是在这些维度中，某个输入要么都发生了广播，要么都不需要广播。例如：
    ///   > {2, 3, 4, 5, 6} -> {6, 4, 30}
    ///   > {2, 3, 1, 5, 6}
    ///
    /// 为了实现这些优化，广播器维护和比较两个布尔向量，记录当前维度的广播状态是否变化。
    /// 所有输入在某个维度的步长会在这个维度确定下来时计算和保存。
    Broadcaster::Broadcaster(std::vector<slice_t<dim_t>> inputs) noexcept
        : strides{}, outputsCount(1), inputsCount(inputs.size()) {
        ASSERT(inputsCount > 0, "Broadcaster: no inputs");

        std::vector<bool>
            broadcastState(inputsCount, false),
            broadcastNext(inputsCount);
        std::vector<dim_t> muls(inputsCount, 1);
        while (true) {
            dim_t shape = 1;
            {
                auto allEnd = true;
                broadcastNext.assign(inputsCount, false);
                for (auto i : range0_(inputsCount)) {
                    // out of dimension for this input
                    if (inputs[i].end_ == inputs[i].begin_) { continue; }
                    // input dimension is not 1
                    if (auto dim = *--inputs[i].end_; broadcastNext[i] = dim != 1) {
                        if (shape == 1) {
                            shape = dim;
                        } else {
                            ASSERT(shape == dim, "Broadcaster: shape mismatch");
                        }
                    }
                    // not all inputs are exhausted for this dimension
                    allEnd = false;
                }
                if (allEnd) { break; }
                if (shape == 1) { continue; }
            }
            if (broadcastNext != broadcastState) {
                broadcastState = std::move(broadcastNext);
                strides.resize(strides.size() + inputsCount + 1, 0);

                auto itRev = strides.rbegin();
                for (auto i : range0_(inputsCount)) {
                    if (broadcastState[i]) {
                        *itRev = muls[i];
                        muls[i] *= shape;
                    }
                    ++itRev;
                }
                *itRev = outputsCount;
            } else {
                for (auto i : range0_(inputsCount)) {
                    if (broadcastState[i]) {
                        muls[i] *= shape;
                    }
                }
            }
            outputsCount *= shape;
        }
        std::reverse(strides.begin(), strides.end());
    }

    Broadcaster::Broadcaster(TensorRefs const &inputs) noexcept
        : Broadcaster([&] {
              std::vector<slice_t<dim_t>> ans(inputs.size());
              std::transform(inputs.begin(), inputs.end(), ans.begin(),
                             [](auto const &t) { return slice(t.get().shape.data(), t.get().rank()); });
              return ans;
          }()) {}

    void Broadcaster::locate(dim_t k, dim_t ans[]) const noexcept {
        long rem = k;
        std::fill_n(ans, inputsCount, 0);
        for (auto i : range0_(strides.size() / (inputsCount + 1))) {
            auto dim = strides.data() + (inputsCount + 1) * i;
            auto div = std::div(rem, dim[inputsCount]);
            for (auto j : range0_(inputsCount)) { ans[j] += dim[j] * div.quot; }
            rem = div.rem;
        }
    }

}// namespace refactor::kernel
