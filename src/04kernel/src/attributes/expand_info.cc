#include "kernel/attributes/expand_info.h"
#include <numeric>

namespace refactor::kernel {

    bool ExpandInfo::Dim::operator==(Dim const &rhs) const noexcept {
        return i == rhs.i && o == rhs.o;
    }
    bool ExpandInfo::Dim::operator!=(Dim const &rhs) const noexcept {
        return !operator==(rhs);
    }

    ExpandInfo::ExpandInfo(
        DataType dataType,
        slice_t<dim_t> input,
        slice_t<dim_t> output) noexcept
        : strides{{1, 1}},
          blockCount(1),
          blockSize(dataType.size()) {
        ASSERT(input.size() <= output.size(), "Unreachable");
        dim_t stride = 1;
        for (auto i = input.end_,
                  o = output.end_;
             o != output.begin_;) {
            auto i_ = i == input.begin_ ? 1 : *--i,
                 o_ = *--o;
            if (o_ == 1) { continue; }
            if (auto &it = strides.back(); i_ == 1) {
                if (it.i) { strides.push_back({0, blockCount}); }
            } else {
                if (!it.i) { strides.push_back({stride, blockCount}); }
                stride *= i_;
            }
            blockCount *= o_;
        }
        if (strides.size() == 1) {
            // 没有发生广播
            blockSize *= blockCount;
            blockCount = 1;
            strides = {};
            return;
        }
        std::reverse(strides.begin(), strides.end());
        strides.pop_back();
        strides.shrink_to_fit();

        auto tail = strides.back();
        ASSERT(tail.i == 0, "Unreachable");

        blockSize *= tail.o;
        blockCount /= tail.o;
        for (auto &s : strides) {
            s.i /= tail.o;
            s.o /= tail.o;
        }
    }

    ExpandInfo::ExpandInfo(
        Tensor const &input,
        Tensor const &output) noexcept
        : ExpandInfo(input.dataType,
                     slice(input.shape.data(), input.rank()),
                     slice(output.shape.data(), output.rank())) {}

    ExpandInfo ExpandInfo::reform(dim_t maxblockSize) const noexcept {
        auto ans = *this;
        ans.reformAssign(maxblockSize);
        return ans;
    }
    void ExpandInfo::reformAssign(dim_t maxblockSize) noexcept {
        if (strides.empty()) { return; }
        auto blockSize_ = std::gcd(blockSize, maxblockSize);
        if (blockSize_ == blockSize) { return; }
        auto times = blockSize / blockSize_;
        blockCount *= times;
        blockSize = blockSize_;
        for (auto &s : strides) {
            s.i *= times;
            s.o *= times;
        }
        strides.resize(strides.size() + 1);
        strides.rbegin()[0] = {1, 1};
    }

}// namespace refactor::kernel
